# Weather RAG Explorer

A command-line tool for exploring weather data using AWS Bedrock and RAG (Retrieval-Augmented Generation) technology. This application enables natural language querying of weather datasets, providing AI-generated insights powered by Claude.

## Features

- Natural language queries for weather data analysis
- Semantic search using vector embeddings
- AI-powered responses using Claude models
- Support for local CSV files and NOAA precipitation datasets
- Comprehensive AWS Bedrock access diagnostics

## Prerequisites

- Python 3.8+
- AWS account with Bedrock access enabled
- AWS credentials configured (via AWS CLI or environment variables)
- AWS Bedrock model subscriptions for:
  - An embedding model (e.g., Amazon Titan Embeddings)
  - A language model (e.g., Claude)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/weather-rag-explorer.git
   cd weather-rag-explorer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure AWS credentials:
   ```bash
   aws configure
   ```

## Usage

### Check AWS Bedrock Access

Before running the main application, verify your AWS Bedrock access:

```bash
python weather_rag_cli.py --check-only
```

This will check if you have access to the necessary AWS Bedrock models without loading any data.

### Run with Default Settings

```bash
python weather_rag_cli.py
```

This will analyze precipitation data for 2022 from NOAA's public dataset.

### Analyze a Specific Year

```bash
python weather_rag_cli.py --year 2021
```

### Use a Local CSV File

```bash
python weather_rag_cli.py --local-file path/to/your/weather_data.csv
```

The CSV file should have at least one of these columns: DATE, TMAX, TMIN, PRCP.

## Example Queries

Once the application is running, you can ask questions like:

- "What was the average precipitation in July?"
- "Which month had the highest temperatures?"
- "Was there a noticeable warming trend over the course of the year?"
- "Compare precipitation patterns between summer and winter."
- "What were the top 5 hottest days of the year?"

## AWS Bedrock Model Access

The application requires access to:

1. An embedding model (one of):
   - amazon.titan-embed-text-v2:0
   - amazon.titan-embed-g1-text-02
   - amazon.titan-multimodal-embed-g1:0
   - amazon.titan-embed-text-v1
   - cohere.embed-english-v3
   - cohere.embed-multilingual-v3

2. A language model (one of):
   - anthropic.claude-3-7-sonnet-20250219-v1:0
   - anthropic.claude-3-5-haiku-20241022-v1:0
   - anthropic.claude-3-sonnet-20240229-v1:0
   - anthropic.claude-3-haiku-20240307-v1:0

You can request access to these models in the AWS Bedrock console under "Model access".

## AWS Region Configuration

By default, the application respects the AWS region configuration in your AWS profile. If you have a preferred region, set it with:

```bash
aws configure set region us-west-2
```

## Troubleshooting

If you encounter issues:

1. Run `--check-only` to verify model access
2. Make sure your AWS credentials are configured
3. Check that you have requested access to the models in the AWS Bedrock console
4. Verify your AWS region settings with `aws configure get region`
5. Ensure boto3 is updated to at least version 1.28.0

## License

MIT

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.