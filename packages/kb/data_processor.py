#!/usr/bin/env python3
"""
AgriMind Knowledge Base Data Processor
Processes farming data specifically for West Bengal and Kolkata
"""

import pandas as pd
import PyPDF2
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgriDataProcessor:
    def __init__(self, kb_path: str = "./"):
        self.kb_path = Path(kb_path)
        self.output_path = self.kb_path / "processed"
        self.output_path.mkdir(exist_ok=True)
        
    def process_market_data(self, csv_file: str) -> Dict[str, Any]:
        """Process West Bengal and Kolkata market data from CSV"""
        logger.info(f"Processing market data from {csv_file}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Filter for West Bengal data
            wb_data = df[df['State'] == 'West Bengal'].copy()
            
            # Filter specifically for Kolkata
            kolkata_data = wb_data[wb_data['District'] == 'Kolkata'].copy()
            
            # Clean and process the data
            processed_data = {
                'west_bengal_summary': self._summarize_market_data(wb_data),
                'kolkata_summary': self._summarize_market_data(kolkata_data),
                'districts': list(wb_data['District'].unique()),
                'commodities': list(wb_data['Commodity'].unique()),
                'markets': list(wb_data['Market'].unique()),
                'price_analysis': self._analyze_prices(wb_data),
                'kolkata_markets': self._get_kolkata_market_details(kolkata_data),
                'district_wise_data': self._group_by_district(wb_data)
            }
            
            # Save processed data
            output_file = self.output_path / "west_bengal_market_data.json"
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2, default=str)
            
            # Create separate CSV files for different districts
            self._create_district_csvs(wb_data)
            
            logger.info(f"Market data processed successfully. Output saved to {output_file}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return {}
    
    def _summarize_market_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics for market data"""
        if df.empty:
            return {}
            
        return {
            'total_records': len(df),
            'unique_districts': len(df['District'].unique()),
            'unique_commodities': len(df['Commodity'].unique()),
            'unique_markets': len(df['Market'].unique()),
            'date_range': {
                'start': df['Arrival_Date'].min(),
                'end': df['Arrival_Date'].max()
            },
            'price_summary': {
                'avg_min_price': df['Min_x0020_Price'].mean(),
                'avg_max_price': df['Max_x0020_Price'].mean(),
                'avg_modal_price': df['Modal_x0020_Price'].mean()
            }
        }
    
    def _analyze_prices(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trends and patterns"""
        price_analysis = {}
        
        # Top 10 most expensive commodities by modal price
        top_expensive = df.nlargest(10, 'Modal_x0020_Price')[['Commodity', 'Variety', 'Modal_x0020_Price', 'District', 'Market']]
        price_analysis['top_expensive_commodities'] = top_expensive.to_dict('records')
        
        # Top 10 cheapest commodities
        top_cheap = df.nsmallest(10, 'Modal_x0020_Price')[['Commodity', 'Variety', 'Modal_x0020_Price', 'District', 'Market']]
        price_analysis['top_cheap_commodities'] = top_cheap.to_dict('records')
        
        # Average prices by commodity
        avg_prices = df.groupby('Commodity')['Modal_x0020_Price'].agg(['mean', 'min', 'max', 'count']).round(2)
        price_analysis['commodity_price_stats'] = avg_prices.to_dict('index')
        
        # District-wise average prices
        district_prices = df.groupby('District')['Modal_x0020_Price'].agg(['mean', 'count']).round(2)
        price_analysis['district_price_stats'] = district_prices.to_dict('index')
        
        return price_analysis
    
    def _get_kolkata_market_details(self, kolkata_df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed information about Kolkata markets"""
        if kolkata_df.empty:
            return {}
            
        markets = {}
        for market in kolkata_df['Market'].unique():
            market_data = kolkata_df[kolkata_df['Market'] == market]
            markets[market] = {
                'commodities': market_data['Commodity'].unique().tolist(),
                'commodity_count': len(market_data['Commodity'].unique()),
                'avg_price': market_data['Modal_x0020_Price'].mean(),
                'price_range': {
                    'min': market_data['Min_x0020_Price'].min(),
                    'max': market_data['Max_x0020_Price'].max()
                },
                'records': market_data.to_dict('records')
            }
        
        return markets
    
    def _group_by_district(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Group data by district for easy access"""
        districts = {}
        for district in df['District'].unique():
            district_data = df[df['District'] == district]
            districts[district] = {
                'markets': district_data['Market'].unique().tolist(),
                'commodities': district_data['Commodity'].unique().tolist(),
                'total_records': len(district_data),
                'avg_modal_price': district_data['Modal_x0020_Price'].mean(),
                'data': district_data.to_dict('records')
            }
        
        return districts
    
    def _create_district_csvs(self, df: pd.DataFrame):
        """Create separate CSV files for each district"""
        district_path = self.output_path / "districts"
        district_path.mkdir(exist_ok=True)
        
        for district in df['District'].unique():
            district_data = df[df['District'] == district]
            filename = f"{district.replace(' ', '_').lower()}_market_data.csv"
            district_data.to_csv(district_path / filename, index=False)
            logger.info(f"Created {filename} with {len(district_data)} records")
    
    def process_pdf_documents(self, pdf_dir: str = "others") -> Dict[str, Any]:
        """Process PDF documents and extract text content"""
        pdf_path = self.kb_path / pdf_dir
        processed_pdfs = {}
        
        for pdf_file in pdf_path.glob("*.pdf"):
            logger.info(f"Processing PDF: {pdf_file.name}")
            
            try:
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = ""
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text_content += page.extract_text() + "\n"
                    
                    processed_pdfs[pdf_file.stem] = {
                        'filename': pdf_file.name,
                        'page_count': len(pdf_reader.pages),
                        'content': text_content,
                        'summary': self._generate_pdf_summary(text_content, pdf_file.name),
                        'processed_date': datetime.now().isoformat()
                    }
                    
                    # Save individual PDF content
                    pdf_output_file = self.output_path / f"{pdf_file.stem}_content.txt"
                    with open(pdf_output_file, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                        
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_file.name}: {e}")
                processed_pdfs[pdf_file.stem] = {
                    'filename': pdf_file.name,
                    'error': str(e),
                    'processed_date': datetime.now().isoformat()
                }
        
        # Save all PDF processing results
        pdf_output_file = self.output_path / "pdf_processing_results.json"
        with open(pdf_output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_pdfs, f, indent=2, ensure_ascii=False)
            
        logger.info(f"PDF processing completed. Results saved to {pdf_output_file}")
        return processed_pdfs
    
    def _generate_pdf_summary(self, content: str, filename: str) -> Dict[str, Any]:
        """Generate a summary of PDF content"""
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Look for West Bengal or Kolkata mentions
        wb_mentions = len([line for line in non_empty_lines if 'west bengal' in line.lower()])
        kolkata_mentions = len([line for line in non_empty_lines if 'kolkata' in line.lower()])
        
        # Look for agricultural terms
        agri_terms = ['crop', 'farming', 'agriculture', 'cultivation', 'harvest', 'yield', 'farmer', 'kharif', 'rabi']
        agri_mentions = sum([len([line for line in non_empty_lines if term in line.lower()]) for term in agri_terms])
        
        return {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'character_count': len(content),
            'west_bengal_mentions': wb_mentions,
            'kolkata_mentions': kolkata_mentions,
            'agriculture_related_mentions': agri_mentions,
            'is_agriculture_related': agri_mentions > 5,
            'is_west_bengal_specific': wb_mentions > 0 or kolkata_mentions > 0
        }
    
    def create_knowledge_base_index(self) -> Dict[str, Any]:
        """Create an index of all processed knowledge base content"""
        index = {
            'metadata': {
                'created_date': datetime.now().isoformat(),
                'focus_region': 'West Bengal and Kolkata',
                'data_types': ['market_prices', 'agricultural_reports', 'government_advisories']
            },
            'market_data': {},
            'pdf_documents': {},
            'recommendations': self._generate_recommendations()
        }
        
        # Load market data if available
        market_file = self.output_path / "west_bengal_market_data.json"
        if market_file.exists():
            with open(market_file, 'r') as f:
                index['market_data'] = json.load(f)
        
        # Load PDF processing results if available
        pdf_file = self.output_path / "pdf_processing_results.json"
        if pdf_file.exists():
            with open(pdf_file, 'r', encoding='utf-8') as f:
                index['pdf_documents'] = json.load(f)
        
        # Save the index
        index_file = self.output_path / "knowledge_base_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Knowledge base index created: {index_file}")
        return index
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for RAG system optimization"""
        return [
            "Focus on seasonal crop patterns specific to West Bengal climate",
            "Include local variety names and regional terminology",
            "Incorporate district-wise agricultural practices and soil types",
            "Add weather pattern analysis for monsoon impact on farming",
            "Include government scheme information specific to West Bengal farmers",
            "Consider local market integration and supply chain data",
            "Add crop disease and pest management specific to the region",
            "Include irrigation and water management practices for the region"
        ]


def main():
    """Main function to run the data processing"""
    processor = AgriDataProcessor()
    
    # Process the market data CSV
    csv_file = "others/9ef84268-d588-465a-a308-a864a43d0070.csv"
    if os.path.exists(csv_file):
        processor.process_market_data(csv_file)
    else:
        logger.warning(f"CSV file not found: {csv_file}")
    
    # Process PDF documents
    processor.process_pdf_documents()
    
    # Create knowledge base index
    processor.create_knowledge_base_index()
    
    logger.info("Data processing completed successfully!")


if __name__ == "__main__":
    main()
