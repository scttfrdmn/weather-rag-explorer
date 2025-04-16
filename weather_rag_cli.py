#!/usr/bin/env python3

import os
import sys
import click
import boto3
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Langchain and AI components
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Bedrock

class WeatherDataRAGExplorer:
    def __init__(self, year: int = 2022):
        """
        Initialize AWS Bedrock client and RAG components
        
        Args:
            year (int): Year of weather data to load
        """
        # AWS Bedrock Configuration
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime', 
            region_name='us-west-2'  # Adjust your preferred region
        )
        
        # AWS S3 Client for Open Data Archive
        self.s3_client = boto3.client('s3')
        
        # Selected year
        self.selected_year = year
        
        # Embedding and LLM Models
        self.embedding_model = BedrockEmbeddings(
            client=self.bedrock_runtime,
            model_id="amazon.titan-embed-text-v1"
        )
        
        self.llm = Bedrock(
            client=self.bedrock_runtime,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0"
        )
        
        # Vector Store
        self.vector_store = None
        self.weather_data = None
    
    def fetch_weather_data(self, dataset: str = "daily") -> pd.DataFrame:
        """
        Fetch weather data from AWS Open Data Archive
        
        Args:
            dataset (str): Name of the weather dataset
        
        Returns:
            pd.DataFrame: Retrieved weather data
        """
        try:
            # Example bucket and key structure - adjust based on actual AWS Open Data setup
            bucket = 'noaa-ghcn-pds'
            key = f'daily/{self.selected_year}.csv'
            
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            self.weather_data = pd.read_csv(response['Body'], low_memory=False)
            
            return self.weather_data
        
        except Exception as e:
            click.echo(f"Error fetching weather data: {e}")
            return pd.DataFrame()
    
    def prepare_retrieval_index(self):
        """
        Prepare vector store for RAG retrieval
        """
        if self.weather_data is None:
            click.echo("Please fetch weather data first.")
            return False
        
        # Convert DataFrame to text for embedding
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Create text chunks
        texts = [
            f"Date: {row['DATE']}, "
            f"Station: {row['STATION']}, "
            f"Max Temp: {row.get('TMAX', 'N/A')}, "
            f"Min Temp: {row.get('TMIN', 'N/A')}, "
            f"Precipitation: {row.get('PRCP', 'N/A')}"
            for _, row in self.weather_data.iterrows()
        ]
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            texts, 
            self.embedding_model
        )
        
        return True
    
    def query_weather_data(self, query: str) -> str:
        """
        Perform RAG-enhanced query on weather data
        
        Args:
            query (str): User's natural language query
        
        Returns:
            str: RAG-enhanced response
        """
        if self.vector_store is None:
            return "Please load and prepare weather data first."
        
        retriever = self.vector_store.as_retriever()
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        response = rag_chain.invoke(query)
        return response['result']

@click.command()
@click.option('--year', default=2022, help='Year of weather data to analyze')
def cli(year):
    """
    Weather Data RAG Explorer CLI
    
    Interactive CLI for exploring historical weather data using RAG technology
    """
    # Clear screen for a clean interface
    os.system('cls' if os.name == 'nt' else 'clear')
    
    click.echo(click.style("""
    🌦️ Weather Data RAG Explorer 🌍
    -----------------------------
    An AI-powered weather data exploration tool
    """, fg='green', bold=True))
    
    # Initialize the explorer
    try:
        explorer = WeatherDataRAGExplorer(year)
        
        # Fetch weather data
        click.echo(f"\nFetching weather data for {year}...")
        weather_data = explorer.fetch_weather_data()
        
        if weather_data.empty:
            click.echo(click.style("Failed to fetch weather data. Exiting.", fg='red'))
            sys.exit(1)
        
        # Prepare retrieval index
        click.echo("Preparing semantic search index...")
        if not explorer.prepare_retrieval_index():
            click.echo(click.style("Failed to prepare data index. Exiting.", fg='red'))
            sys.exit(1)
        
        # Interactive query loop
        while True:
            query = click.prompt(
                click.style("\n❓ Ask a question about the weather data", fg='blue') + 
                click.style("\n(or type 'exit' to quit)", fg='yellow')
            )
            
            # Exit condition
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            # Process query
            try:
                click.echo("\n🤖 AI Response:")
                response = explorer.query_weather_data(query)
                click.echo(click.style(response, fg='green'))
            
            except Exception as e:
                click.echo(click.style(f"Error processing query: {e}", fg='red'))
        
        # Farewell message
        click.echo(click.style("\nThank you for using Weather Data RAG Explorer!", fg='green'))
    
    except Exception as e:
        click.echo(click.style(f"An error occurred: {e}", fg='red'))
        sys.exit(1)

if __name__ == "__main__":
    cli()