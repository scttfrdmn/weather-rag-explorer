#!/usr/bin/env python3

"""
AWS Bedrock Access Checker

This script checks if your AWS credentials have proper access to Bedrock services
and the necessary models for the Weather RAG Explorer to function correctly.
"""

import os
import sys
import boto3

# Try importing from langchain packages
try:
    from langchain_community.embeddings import BedrockEmbeddings
    from langchain_aws import ChatBedrock
except ImportError:
    print("❌ Error: Required LangChain packages not found.")
    print("Please install them with: pip install langchain-community langchain-aws")
    sys.exit(1)

def main():
    """
    Main function to check AWS Bedrock access.
    """
    print("\n=================================================")
    print("🛠️  AWS Bedrock Access Checker for Weather RAG  🛠️")
    print("=================================================")
    print("This tool will verify your AWS credentials and Bedrock model access")
    print("=================================================\n")
    
    # Check AWS credentials
    print("\n--- AWS Credentials Check ---")
    
    try:
        # Create a session to verify credentials
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials is None:
            print("❌ No AWS credentials found!")
            print("Please run 'aws configure' or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
            return False
        
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
    
    except Exception as e:
        print(f"❌ AWS Credentials Error: {e}")
        print("Please configure your AWS credentials properly")
        return False
    
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
        print("Please check your network connectivity and AWS permissions")
        return False
    
    # Check available models
    print("\n--- AWS Bedrock Model Check ---")
    bedrock_client = boto3.client('bedrock', region_name=bedrock_runtime.meta.region_name)
    models_response = bedrock_client.list_foundation_models()
    
    # Print available models
    print("Available Bedrock Models:")
    available_models = []
    for model in models_response.get('modelSummaries', []):
        model_id = model.get('modelId')
        available_models.append(model_id)
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
    embedding_available = False
    for model_id in embedding_models:
        if model_id in available_models:
            print(f"✓ Embedding model {model_id} is available in your account")
            embedding_available = True
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
        print("RAG cannot function without embeddings! Please subscribe to at least one embedding model.")
        if embedding_available:
            print("You have model access but there's a runtime error. Check your AWS permissions.")
        else:
            print("You need to subscribe to embedding models in the AWS Bedrock console:")
            print("Visit: https://console.aws.amazon.com/bedrock/home#/modelaccess")
        return False
    
    # Check LLM models
    print("\n--- Language Model Check ---")
    llm_models = [
        # Models that support on-demand throughput
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-instant-v1",
        "amazon.titan-text-express-v1"
    ]
    
    llm_available = False
    for model_id in llm_models:
        if model_id in available_models:
            print(f"✓ Language model {model_id} is available in your account")
            llm_available = True
        else:
            print(f"❌ Language model {model_id} is NOT available in your account")
    
    llm = None
    for model_id in llm_models:
        if model_id in available_models:
            print(f"Trying to connect to language model: {model_id}")
            try:
                # Add anthropic_version for Anthropic models
                model_kwargs = {"temperature": 0.7, "max_tokens": 1000}
                
                if "anthropic" in model_id:
                    model_kwargs["anthropic_version"] = "bedrock-2023-05-31"
                    
                llm = ChatBedrock(
                    client=bedrock_runtime,
                    model_id=model_id,
                    model_kwargs=model_kwargs
                )
                print(f"✓ Successfully connected to language model: {model_id}")
                break
            except Exception as e:
                print(f"❌ Failed to use language model {model_id}: {e}")
    
    if llm is None:
        print("\n⚠️ Warning: No language models are fully accessible.")
        if llm_available:
            print("You have model access but there's a runtime error. Check your AWS permissions.")
        else:
            print("For AI-enhanced responses, subscribe to at least one LLM model in the AWS Bedrock console")
        
        # Not a fatal error, can run with embeddings only
        print("⚠️ The application can still run with limited functionality (embeddings only).")
    
    # Final summary
    print("\n=================================================")
    if embedding_model and llm:
        print("✅ FULL AWS BEDROCK ACCESS VERIFIED")
        print("You have access to both embedding and language models.")
        print("The Weather RAG Explorer will work with full functionality.")
        return True
    elif embedding_model:
        print("⚠️ PARTIAL AWS BEDROCK ACCESS VERIFIED")
        print("You have access to embedding models but not language models.")
        print("The Weather RAG Explorer will work with limited AI capabilities.")
        return True
    else:
        print("❌ INSUFFICIENT AWS BEDROCK ACCESS")
        print("You need at least one embedding model to run the Weather RAG Explorer.")
        print("Please subscribe to the required models and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)