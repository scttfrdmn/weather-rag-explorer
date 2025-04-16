# Weather Data RAG Explorer

## Project Overview

A CLI-based weather data exploration tool that uses Retrieval-Augmented Generation (RAG) to provide intelligent insights into historical climate data.

## 🌐 Key Features

- AI-powered natural language querying
- Access to NOAA Global Historical Climatology Network (GHCN) Daily Dataset
- Semantic search across historical weather records
- Interactive CLI interface

## 🚀 Prerequisites

- Python 3.8+
- AWS Account
- AWS Bedrock Access
- AWS CLI Configured

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/weather-rag-explorer.git
cd weather-rag-explorer
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 4. Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure

# Ensure you have the necessary Bedrock permissions
```

## 🖥️ Running the Application

```bash
# Run the CLI application for the current year (2022 by default)
python weather_rag_cli.py

# Specify a different year
python weather_rag_cli.py --year 2020
```

## 🌟 Example Queries

- "What were the most extreme temperature days in Phoenix during 2022?"
- "Describe precipitation patterns this year"
- "Find the hottest and coldest days"
- "Compare summer temperatures across different stations"

## 🔍 How It Works

The application uses:
- AWS Bedrock Claude 3 for natural language processing
- FAISS for semantic vector search
- NOAA's historical weather dataset

## 🚧 Limitations

- Requires AWS Bedrock subscription
- Data availability depends on NOAA dataset
- Semantic search accuracy varies with query complexity

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- NOAA for the GHCN Daily Dataset
- Anthropic for Claude 3
- AWS for Bedrock AI services