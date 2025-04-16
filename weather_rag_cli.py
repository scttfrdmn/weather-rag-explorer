#!/usr/bin/env python3

import os
import sys
import click
import boto3
import pandas as pd
import numpy as np
import faiss
import netCDF4
import xarray as xr
import s3fs
from typing import List, Dict, Any

# Modern imports to replace pkg_resources
try:
    # Python 3.8+
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

# Rich for enhanced progress and styling
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Updated Langchain imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Version comparison utility
def is_version_less_than(v1, v2):
    """Compare two version strings"""
    try:
        v1_parts = [int(x) for x in v1.split('.')]
        v2_parts = [int(x) for x in v2.split('.')]
        
        for i in range(min(len(v1_parts), len(v2_parts))):
            if v1_parts[i] < v2_parts[i]:
                return True
            elif v1_parts[i] > v2_parts[i]:
                return False
        
        # If we get here, the common parts are equal, so the shorter version is less
        return len(v1_parts) < len(v2_parts)
    except (ValueError, AttributeError):
        # If comparison fails, assume versions are compatible
        return False

class WeatherDataRAGExplorer:
    def __init__(self, year: int = 2022, local_file: str = None):
        """
        Initialize AWS Bedrock client and RAG components
        
        Args:
            year (int): Year of weather data to load
            local_file (str): Path to local CSV file
        """
        # Rich console for enhanced output
        self.console = Console()
        
        # Selected year
        self.selected_year = year
        
        # Initialize Bedrock components
        self.bedrock_runtime, self.embedding_model, self.llm = self._initialize_bedrock_components()
        
        # Prepare sample data if no other source is available
        self.sample_data = self._generate_sample_data()
        
        # Try to fetch or load data
        self.weather_data = self._load_weather_data(local_file)
        
        # FAISS Specific Attributes
        self.faiss_index = None
        self.embeddings = None

    def _parse_netcdf_filename_date(self, filename):
        """
        Extract date from NetCDF filename in format gpcp_v01r03_daily_d20000121_c20170530.nc
        
        Args:
            filename (str): Filename to parse
        
        Returns:
            str: Extracted date in YYYY-MM-DD format
        """
        try:
            # Extract date portion (d20000121)
            date_match = filename.split('_')[3]
            
            # Remove the leading 'd'
            date_str = date_match[1:]
            
            # Parse into YYYY, MM, DD
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            
            return f"{year}-{month}-{day}"
        except Exception as e:
            print(f"Warning: Error parsing filename {filename}: {e}")
            return None
    
    def _initialize_bedrock_components(self):
        """
        Initialize AWS Bedrock components with robust error handling
        using modern importlib.metadata instead of pkg_resources
        
        Returns:
            tuple: (bedrock_runtime, embedding_model, llm)
        
        Raises:
            Exception: If embedding models are not available
        """
        try:
            print("\n=== AWS Bedrock Initialization ===")
            
            # Check boto3 version using importlib.metadata
            try:
                boto3_version = version('boto3')
                MIN_BOTO3_VERSION = "1.28.0"
                
                print(f"boto3 version: {boto3_version}")
                
                if is_version_less_than(boto3_version, MIN_BOTO3_VERSION):
                    print(f"⚠️ Warning: Your boto3 version ({boto3_version}) might be too old for Bedrock!")
                    print(f"⚠️ Bedrock requires boto3 >= {MIN_BOTO3_VERSION}")
                
            except PackageNotFoundError:
                print("⚠️ Could not determine boto3 version")
            
            # Check Bedrock regions
            print("\n--- AWS Bedrock Region Check ---")

            # Create a session to get the default region
            session = boto3.Session()
            default_region = session.region_name

            # Create the preferred regions list with the default first (if available)
            preferred_regions = []
            if default_region:
                preferred_regions.append(default_region)
                print(f"✓ Using default region from profile: {default_region}")

            # Add other regions as fallbacks
            fallback_regions = ['us-east-1', 'us-west-2']
            for region in fallback_regions:
                if region not in preferred_regions:
                    preferred_regions.append(region)

            print(f"Order of regions to try: {', '.join(preferred_regions)}")
            
            # Check for AWS credentials
            print("\n--- AWS Credentials Check ---")
            try:
                # Create a session to verify credentials
                session = boto3.Session()
                credentials = session.get_credentials()
                
                if credentials is None:
                    print("❌ No AWS credentials found!")
                    print("Please run 'aws configure' or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
                    raise Exception("No AWS credentials found")
                
                # Print credential source without exposing secrets
                cred_source = "environment variables" if os.environ.get("AWS_ACCESS_KEY_ID") else \
                            "credentials file" if os.path.exists(os.path.expanduser("~/.aws/credentials")) else \
                            "unknown source"
                print(f"✓ Using AWS credentials from: {cred_source}")
                
                # Print identity info if possible
                try:
                    sts = session.client('sts')
                    identity = sts.get_caller_identity()
                    account_id = identity.get('Account', 'unknown')
                    user_id = identity.get('UserId', 'unknown')
                    # Mask for privacy in logs
                    masked_account = f"{account_id[:4]}...{account_id[-4:]}" if len(account_id) > 8 else "****"
                    masked_user = user_id.split('/')[-1][:4] + "***" if '/' in user_id else "****"
                    print(f"✓ AWS Account: {masked_account}")
                    print(f"✓ IAM Identity: {masked_user}")
                except Exception as sts_error:
                    print(f"⚠️ Could not verify AWS identity: {sts_error}")
                
            except Exception as cred_error:
                print(f"❌ AWS Credentials Error: {cred_error}")
                raise Exception(f"AWS credential error: {cred_error}")
            
            # Try each region in order
            bedrock_runtime = None
            for region in preferred_regions:
                print(f"Trying Bedrock in region: {region}")
                try:
                    # Try creating Bedrock client
                    bedrock_runtime = boto3.client(
                        service_name='bedrock-runtime', 
                        region_name=region
                    )
                    
                    # Test if the client works
                    bedrock_client = boto3.client('bedrock', region_name=region)
                    models = bedrock_client.list_foundation_models()
                    model_count = len(models.get('modelSummaries', []))
                    print(f"✓ Connected to Bedrock in {region} - found {model_count} models")
                    break
                    
                except Exception as region_error:
                    print(f"❌ Failed to connect to Bedrock in {region}: {region_error}")
                    bedrock_runtime = None
                    continue
            
            if bedrock_runtime is None:
                print("\n❌ CRITICAL ERROR: Could not connect to AWS Bedrock in any region")
                print("Possible reasons:")
                print("1. Your AWS credentials do not have Bedrock permissions")
                print("2. Your account is not subscribed to AWS Bedrock")
                print("3. AWS Bedrock is not available in your account's regions")
                print("\nTry running 'aws bedrock-agent list-agent-knowledge-bases' to test your permissions")
                raise Exception("Could not connect to AWS Bedrock in any region")
            
            # List available Bedrock models
            print("\n--- AWS Bedrock Model Check ---")
            try:
                bedrock_client = boto3.client('bedrock', region_name=bedrock_runtime.meta.region_name)
                models_response = bedrock_client.list_foundation_models()
                
                # Print available models
                print("Available Bedrock Models:")
                available_models = []
                for model in models_response.get('modelSummaries', []):
                    model_id = model.get('modelId')
                    available_models.append(model_id)
                    # Don't print all models to keep output cleaner
                    # print(f"- {model_id}")
                print(f"Found {len(available_models)} available models")
            except Exception as list_error:
                print(f"❌ Error listing models: {list_error}")
                available_models = []
            
            # Check embedding models
            print("\n--- Embedding Model Check ---")
            embedding_models = [
                # Titan Text Embeddings V2 (access granted)
                "amazon.titan-embed-text-v2:0",
                # Titan Embeddings G1 - Text (access granted)
                "amazon.titan-embed-g1-text-02",
                # Titan Multimodal Embeddings G1 (access granted)
                "amazon.titan-multimodal-embed-g1:0",
                # Original models that may not work
                "amazon.titan-embed-text-v1",
                # Cohere models as backup
                "cohere.embed-english-v3",
                "cohere.embed-multilingual-v3"
            ]
            
            # Show which embedding models are available in the account
            for model_id in embedding_models:
                if model_id in available_models:
                    print(f"✓ Embedding model {model_id} is available in your account")
                else:
                    print(f"❌ Embedding model {model_id} is NOT available in your account")
            
            # Try to connect to embedding models
            embedding_model = None
            for model_id in embedding_models:
                if model_id in available_models:
                    print(f"Trying to connect to embedding model: {model_id}")
                    try:
                        embedding_model = BedrockEmbeddings(
                            client=bedrock_runtime,
                            model_id=model_id
                        )
                        # Test the embedding model with a simple request
                        test_embedding = embedding_model.embed_documents(["Test"])
                        if test_embedding and len(test_embedding) > 0:
                            print(f"✓ Successfully connected to embedding model: {model_id}")
                            break
                    except Exception as e:
                        print(f"❌ Failed to use embedding model {model_id}: {e}")
            
            if embedding_model is None:
                print("\n❌ CRITICAL ERROR: No embedding models available")
                print("RAG cannot function without embeddings!")
                print("Please subscribe to at least one of these embedding models in the AWS Bedrock console:")
                for model_id in embedding_models:
                    print(f"- {model_id}")
                print("\nVisit: https://console.aws.amazon.com/bedrock/home#/modelaccess")
                raise Exception("No embedding models available - RAG cannot function")
            
            # Language Model (with fallback)
            print("\n--- Language Model Check ---")
            llm_models = [
                # Claude models you have access to
                "anthropic.claude-3-7-sonnet-20250219-v1:0",  # Claude 3.7 Sonnet
                "anthropic.claude-3-5-haiku-20241022-v1:0",   # Claude 3.5 Haiku
                
                # Fallback models
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0"
            ]
            
            # Show which language models are available in the account
            for model_id in llm_models:
                if model_id in available_models:
                    print(f"✓ Language model {model_id} is available in your account")
                else:
                    print(f"❌ Language model {model_id} is NOT available in your account")
            
            # Try to connect to language models
            llm = None
            for model_id in llm_models:
                if model_id in available_models:
                    print(f"Trying to connect to language model: {model_id}")
                    try:
                        llm = ChatBedrock(
                            client=bedrock_runtime,
                            model_id=model_id,
                            model_kwargs={"temperature": 0.7, "max_tokens": 1000}
                        )
                        print(f"✓ Successfully connected to language model: {model_id}")
                        break
                    except Exception as e:
                        print(f"❌ Failed to use language model {model_id}: {e}")
            
            if not llm:
                print("\n⚠️ Warning: No language models available.")
                print("RAG responses will be limited to retrieved content only.")
                print("For AI-enhanced responses, subscribe to at least one of these models:")
                for model_id in llm_models:
                    print(f"- {model_id}")
            
            print("\n--- Bedrock Initialization Complete ---")
            return bedrock_runtime, embedding_model, llm
        
        except Exception as e:
            print("\n❌ CRITICAL ERROR: Bedrock Initialization Failed")
            print(f"Error details: {e}")
            
            # Print a traceback for better debugging
            import traceback
            print("\nTraceback:")
            traceback.print_exc()
            
            print("\nPlease resolve the issues above and try again")
            raise Exception(f"Failed to initialize Bedrock components: {e}")
    
    def _generate_sample_data(self):
        """
        Generate a sample weather dataset
        
        Returns:
            pd.DataFrame: Sample weather data
        """
        print("Generating sample weather data...")
        sample_data = pd.DataFrame({
            'DATE': pd.date_range(start=f'{self.selected_year}-01-01', 
                                  end=f'{self.selected_year}-12-31', 
                                  freq='D'),
            'STATION': 'SAMPLE_STATION',
            'TMAX': np.random.randint(50, 100, size=365),  # Daily high temps
            'TMIN': np.random.randint(20, 70, size=365),  # Daily low temps
            'PRCP': np.random.randint(0, 50, size=365)    # Daily precipitation
        })
        return sample_data
    
    def _load_weather_data(self, local_file: str = None):
        """
        Load weather data from various sources
        
        Args:
            local_file (str): Path to local CSV file
        
        Returns:
            pd.DataFrame: Loaded weather data
        """
        print("\n=== Loading Weather Data ===")
        
        # Try local file first
        if local_file and os.path.exists(local_file):
            try:
                print(f"Loading data from local file: {local_file}")
                return pd.read_csv(local_file)
            except Exception as e:
                print(f"Error loading local file: {e}")
        
        # Try AWS S3 precipitation data
        try:
            print("Loading data from AWS S3...")
            return self._fetch_aws_s3_data()
        except Exception as e:
            print(f"AWS S3 data fetch failed: {e}")
        
        # Fallback to sample data
        print("Using generated sample weather data")
        return self._generate_sample_data()

    def _fetch_aws_s3_data(self):
        """
        Fetch precipitation data from AWS S3 Open Data Registry
        
        Returns:
            pd.DataFrame: Converted weather data
        """
        import boto3
        import s3fs
        import xarray as xr
        import pandas as pd
        import numpy as np
        import io
        
        # Create S3 client to list bucket contents
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        try:
            # List objects in the bucket to understand its structure
            print(f"Listing objects in S3 bucket for year {self.selected_year}...")
            response = s3_client.list_objects_v2(
                Bucket='noaa-cdr-precip-gpcp-daily-pds', 
                Prefix=f'data/{self.selected_year}/'
            )
            
            # Print available objects for debugging
            print("Available objects in precipitation bucket:")
            available_files = []
            matching_files = []
            for obj in response.get('Contents', []):
                file_key = obj['Key']
                available_files.append(file_key)
                
                # Check if file matches the year
                if f'_{self.selected_year}' in file_key or f'd{self.selected_year}' in file_key:
                    matching_files.append(file_key)
                
                print(f"- {file_key}")
            
            # Create S3 filesystem with no sign request
            s3 = s3fs.S3FileSystem(anon=True)  # Anonymous access
            
            # If specific matching files found, use those
            if matching_files:
                data_sources = [f's3://noaa-cdr-precip-gpcp-daily-pds/{f}' for f in matching_files]
            else:
                # Fallback to general pattern
                data_sources = [
                    f's3://noaa-cdr-precip-gpcp-daily-pds/data/{self.selected_year}/*.nc',
                    f's3://noaa-cdr-precip-gpcp-daily-pds/*.nc'
                ]
            
            # Collect data from all matching files
            all_data = []
            
            # Outer loop through potential data sources
            for source in data_sources:
                print(f"Attempting to fetch data from {source}")
                
                try:
                    # List matching files
                    matching_files = s3.glob(source)
                    
                    if not matching_files:
                        print(f"No files found matching: {source}")
                        continue
                    
                    # Inner loop through matching files
                    for first_file in matching_files:
                        try:
                            # Skip if file doesn't match the year
                            filename = os.path.basename(first_file)
                            
                            # Check if filename contains the year
                            if str(self.selected_year) not in filename:
                                continue
                            
                            print(f"Processing file: {first_file}")
                            
                            # Read the entire file content
                            with s3.open(first_file, 'rb') as f:
                                file_content = f.read()
                            
                            # Create an in-memory bytes buffer
                            buffer = io.BytesIO(file_content)
                            
                            # Try multiple methods to open the file
                            try:
                                # Try to open with h5netcdf engine
                                ds = xr.open_dataset(buffer, engine='h5netcdf')
                            except Exception as netcdf_error:
                                print(f"Failed to open NetCDF file: {netcdf_error}")
                                continue
                            
                            # Try to extract date from filename
                            parsed_date = self._parse_netcdf_filename_date(filename)
                            
                            # Process precipitation data
                            if 'precip' in ds.variables:
                                # Extract precipitation data
                                precip_values = ds['precip'].values
                                
                                # Ensure data is numeric and handle potential issues
                                precip_clean = precip_values.astype(float)
                                
                                # Handle multi-dimensional precipitation data
                                if precip_clean.ndim > 1:
                                    # Flatten and remove any NaN or infinite values
                                    precip_clean = precip_clean.flatten()
                                    precip_clean = precip_clean[np.isfinite(precip_clean)]
                                
                                # Calculate statistics
                                if len(precip_clean) > 0:
                                    df_data = {
                                        'DATE': [parsed_date] if parsed_date else [ds['time'].values[0]],
                                        'PRCP_MEAN': [np.mean(precip_clean)],
                                        'PRCP_MAX': [np.max(precip_clean)],
                                        'PRCP_MIN': [np.min(precip_clean)],
                                        'source_file': [filename]
                                    }
                                    
                                    # Add geographic information if available
                                    if 'latitude' in ds.variables and 'longitude' in ds.variables:
                                        df_data['LATITUDE_MEAN'] = [np.mean(ds['latitude'].values)]
                                        df_data['LONGITUDE_MEAN'] = [np.mean(ds['longitude'].values)]
                                    
                                    # Create DataFrame
                                    df = pd.DataFrame(df_data)
                                    
                                    all_data.append(df)
                            
                        except Exception as file_error:
                            print(f"Error processing file {first_file}: {file_error}")
                            continue
                
                except Exception as source_error:
                    print(f"Error processing source {source}: {source_error}")
                    continue
            
            # Check if any data was collected
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # Sort by date
                combined_df['DATE'] = pd.to_datetime(combined_df['DATE'])
                combined_df = combined_df.sort_values('DATE')
                
                print(f"Successfully retrieved {len(combined_df)} records")
                return combined_df
            
            # If no data was found
            raise ValueError(
                f"No valid precipitation data found for year {self.selected_year}. "
                "Available files: " + ", ".join(available_files) + "\n"
                "Possible reasons:\n"
                "1. Incorrect year or data format\n"
                "2. Changes in S3 bucket structure\n"
                "3. Network or access issues"
            )
        
        except Exception as e:
            print("Error exploring NOAA Precipitation bucket:")
            print(f"{e}")
            raise

    def prepare_embeddings(self):
        """
        Prepare vector embeddings for weather data
        """
        print("\n=== Preparing Vector Embeddings ===")
        
        if self.weather_data is None:
            print("No weather data available.")
            return False
        
        # Report data size
        print(f"Preparing embeddings for {len(self.weather_data)} weather records...")
        
        # Create text representations with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            embedding_task = progress.add_task("[green]Generating embeddings...", total=len(self.weather_data))
            
            # Create text representations
            texts = [
                f"Date: {row['DATE']}, "
                f"Station: {row.get('STATION', 'Unknown')}, "
                f"Max Temp: {row.get('TMAX', 'N/A')}, "
                f"Min Temp: {row.get('TMIN', 'N/A')}, "
                f"Precipitation: {row.get('PRCP', 'N/A')}"
                for _, row in self.weather_data.iterrows()
            ]
            
            # Use Bedrock embeddings
            if self.embedding_model:
                try:
                    self.embeddings = []
                    for text in texts:
                        self.embeddings.append(self.embedding_model.embed_documents([text])[0])
                        progress.update(embedding_task, advance=1)
                    
                    self.embeddings = np.array(self.embeddings)
                    print("Successfully generated embeddings using AWS Bedrock")
                except Exception as e:
                    print(f"Error generating embeddings: {e}")
                    return False
            else:
                # Simple fallback embedding (shouldn't reach here with enhanced error handling)
                print("No embedding model available - using simple fallback")
                self.embeddings = np.random.rand(len(texts), 768)  # 768-dimensional random vectors
            
            # Create FAISS index
            try:
                print("Creating vector search index...")
                dim = self.embeddings.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(self.embeddings)
                
                self.faiss_index = index
                print(f"Successfully created FAISS index with {len(self.embeddings)} vectors")
            except Exception as e:
                print(f"Error creating FAISS index: {e}")
                return False
        
        return True

    def query_weather_data(self, query: str, top_k: int = 5) -> str:
        """
        Perform semantic search and generate RAG response
        
        Args:
            query (str): User's natural language query
            top_k (int): Number of top results to retrieve
        
        Returns:
            str: RAG-enhanced response
        """
        if self.faiss_index is None:
            return "Please load and prepare weather data first."
        
        print(f"\nProcessing query: '{query}'")
        
        # Embed query
        try:
            print("Embedding query...")
            # Use Bedrock embedding
            query_embedding = self.embedding_model.embed_documents([query])[0]
            
            # Search in FAISS index
            print(f"Searching for top {top_k} relevant records...")
            D, I = self.faiss_index.search(
                np.array([query_embedding]), 
                top_k
            )
            
            # Retrieve relevant documents
            retrieved_docs = []
            for i, idx in enumerate(I[0]):
                if idx >= 0 and idx < len(self.weather_data):  # Ensure index is valid
                    record = self.weather_data.iloc[idx]
                    # Format the record nicely
                    record_str = f"Record {i+1}: "
                    for col in record.index:
                        record_str += f"{col}: {record[col]}, "
                    retrieved_docs.append(record_str)
            
            # Prepare context for RAG
            context = "\n".join(retrieved_docs)
            
            # Generate response
            if self.llm:
                print("Generating AI response...")
                try:
                    # Create a prompt template
                    prompt_template = PromptTemplate.from_template(
                        """You are an expert meteorological data analyst. 
                        Given the following context about weather records:
                        
                        {context}
                        
                        Answer the following query with detailed insights:
                        {query}
                        
                        Provide a comprehensive and informative response based on the available data.
                        If the data doesn't contain information to answer the query completely, be clear about what you can and cannot determine from the available records."""
                    )
                    
                    # Create a chain with prompt, LLM, and output parser
                    chain = prompt_template | self.llm | StrOutputParser()
                    
                    # Generate response
                    response = chain.invoke({
                        "context": context,
                        "query": query
                    })
                    
                    return response
                except Exception as e:
                    print(f"Error processing query with LLM: {e}")
                    # Fallback response if LLM fails
                    response = (
                        "Unable to generate AI response due to model access issues. "
                        "Here are the most relevant weather records found:\n\n" + context
                    )
                    return response
            else:
                # Simple fallback response
                print("No LLM available - providing basic response")
                response = (
                    "AI language model not available. "
                    "Here are the most relevant records based on your query:\n\n" + context
                )
                return response
        
        except Exception as e:
            print(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing query: {e}"

@click.command()
@click.option('--year', default=2022, help='Year of weather data to analyze')
@click.option('--local-file', default=None, help='Path to local CSV file')
@click.option('--check-only', is_flag=True, help='Only check Bedrock access without running the application')
def cli(year, local_file, check_only):
    """
    Weather Data RAG Explorer CLI
    
    Interactive CLI for exploring historical weather data using RAG technology
    """
    # Print welcome header immediately to confirm script is running
    print("\n=================================================")
    print("🌦️  Welcome to Weather Data RAG Explorer  🌍")
    print("=================================================")
    print("An AI-powered weather data exploration tool")
    print("=================================================\n")
    
    # Check boto3 version using importlib.metadata
    try:
        boto3_version = version('boto3')
        MIN_BOTO3_VERSION = "1.28.0"
        
        print(f"boto3 version: {boto3_version}")
        
        if is_version_less_than(boto3_version, MIN_BOTO3_VERSION):
            print(f"⚠️ Warning: Your boto3 version ({boto3_version}) might be too old for AWS Bedrock!")
            print(f"⚠️ Recommended version is {MIN_BOTO3_VERSION} or newer.")
            print("⚠️ Upgrade boto3 with: pip install --upgrade boto3")
            
            if not click.confirm("Continue anyway?", default=False):
                print("Exiting. Please upgrade boto3 and try again.")
                sys.exit(1)
    except PackageNotFoundError:
        print("⚠️ Could not determine boto3 version. This may cause issues with AWS Bedrock.")
    
    # If check-only flag is set, just check Bedrock access and exit
    if check_only:
        print("\n=== AWS Bedrock Access Check ===")
        try:
            print("Starting AWS Bedrock access check...")
            
            # Define a separate function for the Bedrock check to avoid loading data
            def check_bedrock_access():
                # Import here to ensure dependencies are available
                import boto3
                from langchain_community.embeddings import BedrockEmbeddings
                from langchain_aws import ChatBedrock
                
                print("\n=== AWS Bedrock Initialization ===")
                
                # Check for AWS credentials
                print("\n--- AWS Credentials Check ---")
                
                # Create a session to verify credentials
                session = boto3.Session()
                credentials = session.get_credentials()
                
                if credentials is None:
                    print("❌ No AWS credentials found!")
                    raise Exception("No AWS credentials found")
                
                # Print credential source without exposing secrets
                cred_source = "environment variables" if os.environ.get("AWS_ACCESS_KEY_ID") else \
                            "credentials file" if os.path.exists(os.path.expanduser("~/.aws/credentials")) else \
                            "unknown source"
                print(f"✓ Using AWS credentials from: {cred_source}")
                
                # Print identity info if possible
                try:
                    sts = session.client('sts')
                    identity = sts.get_caller_identity()
                    account_id = identity.get('Account', 'unknown')
                    user_id = identity.get('UserId', 'unknown')
                    # Mask for privacy in logs
                    masked_account = f"{account_id[:4]}...{account_id[-4:]}" if len(account_id) > 8 else "****"
                    masked_user = user_id.split('/')[-1][:4] + "***" if '/' in user_id else "****"
                    print(f"✓ AWS Account: {masked_account}")
                    print(f"✓ IAM Identity: {masked_user}")
                except Exception as sts_error:
                    print(f"⚠️ Could not verify AWS identity: {sts_error}")
                
                # Check Bedrock regions
                print("\n--- AWS Bedrock Region Check ---")
                bedrock_runtime = None
                
                # Create a session to get the default region
                session = boto3.Session()
                default_region = session.region_name

                # Create the preferred regions list with the default first (if available)
                preferred_regions = []
                if default_region:
                    preferred_regions.append(default_region)
                    print(f"✓ Using default region from profile: {default_region}")

                # Add other regions as fallbacks
                fallback_regions = ['us-east-1', 'us-west-2']
                for region in fallback_regions:
                    if region not in preferred_regions:
                        preferred_regions.append(region)

                print(f"Order of regions to try: {', '.join(preferred_regions)}")
                
                for region in preferred_regions:
                    print(f"Trying Bedrock in region: {region}")
                    try:
                        # Try creating Bedrock client
                        bedrock_runtime = boto3.client(
                            service_name='bedrock-runtime', 
                            region_name=region
                        )
                        
                        # Test if the client works
                        bedrock_client = boto3.client('bedrock', region_name=region)
                        models = bedrock_client.list_foundation_models()
                        model_count = len(models.get('modelSummaries', []))
                        print(f"✓ Connected to Bedrock in {region} - found {model_count} models")
                        break
                        
                    except Exception as region_error:
                        print(f"❌ Failed to connect to Bedrock in {region}: {region_error}")
                
                if bedrock_runtime is None:
                    print("\n❌ Could not connect to AWS Bedrock in any region")
                    raise Exception("Could not connect to AWS Bedrock in any region")
                
                # List available Bedrock models
                print("\n--- AWS Bedrock Model Check ---")
                bedrock_client = boto3.client('bedrock', region_name=bedrock_runtime.meta.region_name)
                models_response = bedrock_client.list_foundation_models()
                
                # Print available models
                print("Available Bedrock Models:")
                available_models = []
                for model in models_response.get('modelSummaries', []):
                    model_id = model.get('modelId')
                    available_models.append(model_id)
                    # Don't print all models to keep output cleaner
                    # print(f"- {model_id}")
                print(f"Found {len(available_models)} available models")
                
                # Check embedding models
                print("\n--- Embedding Model Check ---")
                embedding_models = [
                    # Titan Text Embeddings V2 (access granted)
                    "amazon.titan-embed-text-v2:0",
                    # Titan Embeddings G1 - Text (access granted)
                    "amazon.titan-embed-g1-text-02",
                    # Titan Multimodal Embeddings G1 (access granted)
                    "amazon.titan-multimodal-embed-g1:0",
                    # Original models that may not work
                    "amazon.titan-embed-text-v1",
                    # Cohere models as backup
                    "cohere.embed-english-v3",
                    "cohere.embed-multilingual-v3"
                ]
                
                # Show which embedding models are available in the account
                for model_id in embedding_models:
                    if model_id in available_models:
                        print(f"✓ Embedding model {model_id} is available in your account")
                    else:
                        print(f"❌ Embedding model {model_id} is NOT available in your account")
                
                # Try to connect to embedding models
                embedding_model = None
                for model_id in embedding_models:
                    if model_id in available_models:
                        print(f"Trying to connect to embedding model: {model_id}")
                        try:
                            embedding_model = BedrockEmbeddings(
                                client=bedrock_runtime,
                                model_id=model_id
                            )
                            # Test the embedding model
                            test_embedding = embedding_model.embed_documents(["Test"])
                            if test_embedding and len(test_embedding) > 0:
                                print(f"✓ Successfully connected to embedding model: {model_id}")
                                break
                        except Exception as e:
                            print(f"❌ Failed to use embedding model {model_id}: {e}")
                
                if embedding_model is None:
                    print("\n❌ No embedding models available")
                    raise Exception("No embedding models available - RAG cannot function")
                
                # Find a language model (optional for check)
                print("\n--- Language Model Check ---")
                llm_models = [
                    # Claude models you have access to
                    "anthropic.claude-3-7-sonnet-20250219-v1:0",  # Claude 3.7 Sonnet
                    "anthropic.claude-3-5-haiku-20241022-v1:0",   # Claude 3.5 Haiku
                    
                    # Fallback models
                    "anthropic.claude-3-sonnet-20240229-v1:0",
                    "anthropic.claude-3-haiku-20240307-v1:0"
                ]
                
                llm = None
                for model_id in llm_models:
                    if model_id in available_models:
                        print(f"✓ Language model {model_id} is available in your account")
                        print(f"Trying to connect to language model: {model_id}")
                        try:
                            llm = ChatBedrock(
                                client=bedrock_runtime,
                                model_id=model_id,
                                model_kwargs={"temperature": 0.7, "max_tokens": 1000}
                            )
                            print(f"✓ Successfully connected to language model: {model_id}")
                            break
                        except Exception as e:
                            print(f"❌ Failed to use language model {model_id}: {e}")
                
                return embedding_model, llm
            
            # Run the Bedrock access check
            embedding_model, llm = check_bedrock_access()
            
            # If we reach here, the check is successful
            print("\n✅ AWS Bedrock access check PASSED")
            print("You have the necessary model access to run the application")
            
            # Exit without loading data or continuing to the explorer
            sys.exit(0)
            
        except Exception as e:
            print("\n❌ AWS Bedrock access check FAILED")
            print(f"Error: {e}")
            print("\nPlease fix the issues above and try again.")
            print("For detailed instructions on fixing Bedrock model access:")
            print("1. Go to: https://console.aws.amazon.com/bedrock/home#/modelaccess")
            print("2. Find the embedding models (amazon.titan-embed or cohere.embed)")
            print("3. Request access by selecting them and clicking 'Access model'")
            sys.exit(1)
    
    # Regular operation (not check-only mode)
    try:
        print("\nInitializing Weather Data RAG Explorer...")
        explorer = WeatherDataRAGExplorer(year, local_file)
        
        # Prepare embeddings
        print("\nPreparing vector embeddings for search...")
        if not explorer.prepare_embeddings():
            print("❌ Failed to prepare data index. Exiting.")
            sys.exit(1)
        
        print("\n✅ Initialization complete!\n")
        
        # Interactive query loop
        print("Ask questions about the weather data (type 'exit' to quit)")
        print("Example: What was the average precipitation in July?")
        print("Example: Which month had the highest temperatures?")
        print("-" * 50)
        
        while True:
            query = input("\n❓ Enter your question: ")
            
            # Exit condition
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            # Process query
            try:
                print("\n🤖 AI Response:")
                response = explorer.query_weather_data(query)
                print(response)
                print("-" * 50)
            
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                import traceback
                traceback.print_exc()
        
        # Farewell message
        print("\n✨ Thank you for using Weather Data RAG Explorer! ✨")
    
    except Exception as e:
        print("\n❌ An error occurred during initialization:")
        print(f"Error details: {e}")
        print("\nFor help with AWS Bedrock access issues, run: python check_bedrock_access.py")
        
        # Show full traceback for better debugging
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        
        sys.exit(1)

if __name__ == "__main__":
    cli()