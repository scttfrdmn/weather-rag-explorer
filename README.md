# Weather RAG Explorer

An AI-powered tool for exploring and analyzing historical weather data using AWS Bedrock and RAG (Retrieval Augmented Generation) technology.

## Overview

Weather RAG Explorer leverages AWS Bedrock's foundation models to provide intelligent analysis of weather data. The application can process precipitation data from NOAA's open datasets or from local files, allowing meteorologists, researchers, and weather enthusiasts to query weather patterns using natural language.

## Features

- **Natural Language Queries**: Ask questions about weather data in plain English
- **AWS Bedrock Integration**: Utilizes AWS's powerful foundation models for embedding and analysis
- **NOAA Data Integration**: Connects to NOAA's open weather datasets
- **Vector Search**: Uses FAISS for efficient similarity search of weather records
- **Interactive CLI**: User-friendly command-line interface for data exploration
- **Robust Error Handling**: Comprehensive error detection and fallback mechanisms
- **Comprehensive Analysis Mode**: Analyze all available records for broader questions

## New in This Version

- **Comprehensive Mode**: Now supports analyzing all available records instead of just the top 5 matches
- **Mode Switching**: Toggle between comprehensive and default modes during runtime
- **Statistical Summaries**: Automatically calculates monthly averages and other statistics for large datasets
- **Enhanced Context**: Provides better context for the LLM to generate more thorough responses
- **Improved Prompt Engineering**: Updated prompt template to focus on statistical observations and trends

## Requirements

- Python 3.8 or higher
- AWS account with Bedrock access
- Required Python packages: boto3, pandas, numpy, faiss, netCDF4, xarray, s3fs, click, rich, langchain

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/weather-rag-explorer.git
cd weather-rag-explorer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials:
```bash
aws configure
```

## Usage

### Basic Usage

```bash
python weather_rag_cli.py
```

This will start the application in default mode, which analyzes the top 5 most relevant records for each query.

### Advanced Options

```bash
# Start in comprehensive mode (analyze all records)
python weather_rag_cli.py --use-all-records

# Check Bedrock model access without running the application
python weather_rag_cli.py --check-only

# Specify a different year of data to analyze
python weather_rag_cli.py --year 2021

# Use a local CSV file instead of fetching data from AWS S3
python weather_rag_cli.py --local-file path/to/your/data.csv
```

### Interactive Commands

During the interactive session, you can use the following commands:

- `toggle mode` - Switch between comprehensive mode and default mode
- `exit`, `quit`, or `q` - Exit the application

### Example Questions

Default mode works best with specific questions:
- "What was the precipitation amount on July 3, 2022?"
- "Compare the precipitation levels between July 2-3, 2022."

Comprehensive mode works well with broader questions:
- "Which month had the highest temperatures?"
- "What was the precipitation pattern throughout summer?"
- "Is there a correlation between latitude and precipitation amounts?"

## AWS Bedrock Model Requirements

The application requires access to the following types of AWS Bedrock models:

- **Embedding Models**: At least one of these models must be available
  - amazon.titan-embed-text-v2:0
  - amazon.titan-embed-g1-text-02
  - amazon.titan-multimodal-embed-g1:0
  - cohere.embed-english-v3
  - cohere.embed-multilingual-v3

- **Language Models**: At least one of these models is recommended
  - anthropic.claude-3-sonnet-20240229-v1:0
  - anthropic.claude-3-haiku-20240307-v1:0
  - anthropic.claude-instant-v1
  - amazon.titan-text-express-v1

To enable model access, visit the [AWS Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess) and request access to the required models.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.