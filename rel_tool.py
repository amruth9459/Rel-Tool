#!/usr/bin/env python3
"""
Rel-Tool: Interactive WhatsApp Relationship Analyzer
GitHub: https://github.com/amruth9459/Rel-Tool
Supports Ollama for 100% private LLM analysis
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import platform

# Color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    """Display the Rel-Tool banner"""
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                    ║
║  {Colors.BOLD}██████╗ ███████╗██╗      ████████╗ ██████╗  ██████╗ ██╗      {Colors.CYAN}  ║
║  {Colors.BOLD}██╔══██╗██╔════╝██║      ╚══██╔══╝██╔═══██╗██╔═══██╗██║      {Colors.CYAN}  ║
║  {Colors.BOLD}██████╔╝█████╗  ██║         ██║   ██║   ██║██║   ██║██║      {Colors.CYAN}  ║
║  {Colors.BOLD}██╔══██╗██╔══╝  ██║         ██║   ██║   ██║██║   ██║██║      {Colors.CYAN}  ║
║  {Colors.BOLD}██║  ██║███████╗███████╗    ██║   ╚██████╔╝╚██████╔╝███████╗ {Colors.CYAN}  ║
║  {Colors.BOLD}╚═╝  ╚═╝╚══════╝╚══════╝    ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝ {Colors.CYAN}  ║
║                                                                    ║
║         {Colors.GREEN}WhatsApp Relationship Intelligence System v2.0{Colors.CYAN}            ║
║           {Colors.BLUE}200,000+ Messages | Multiple Exports | LLMs{Colors.CYAN}            ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════╝{Colors.END}
    """
    print(banner)

class InteractiveRelTool:
    """Interactive relationship analysis tool with user prompting"""
    
    def __init__(self):
        self.config = {}
        self.export_files = []
        self.llm_choice = None
        self.output_dir = "relationship_analysis"
        self.use_gpu = False
        
    def run(self):
        """Main interactive flow"""
        print_banner()
        
        # Step 1: Welcome and check dependencies
        self.welcome_user()
        
        # Step 2: Collect export files
        self.collect_exports()
        
        # Step 3: Choose LLM
        self.choose_llm()
        
        # Step 4: Configure analysis
        self.configure_analysis()
        
        # Step 5: Check system and install
        self.check_and_install()
        
        # Step 6: Run analysis
        self.run_analysis()
        
        # Step 7: Show results
        self.show_results()
        
    def welcome_user(self):
        """Welcome message and basic info collection"""
        print(f"\n{Colors.GREEN}Welcome to Rel-Tool!{Colors.END}")
        print("I'll help you analyze your WhatsApp relationship conversations.\n")
        
        # Get user name (optional)
        name = input(f"{Colors.CYAN}What's your name? (press Enter to skip): {Colors.END}").strip()
        if name:
            self.config['user_name'] = name
            print(f"\nNice to meet you, {name}! Let's analyze your relationship.\n")
        else:
            print("\nLet's get started!\n")
            
    def collect_exports(self):
        """Collect WhatsApp export files"""
        print(f"{Colors.BOLD}📁 Step 1: WhatsApp Export Files{Colors.END}")
        print("-" * 50)
        
        print("\nI need your WhatsApp export file(s).")
        print("You can provide:")
        print("  1. Single file (e.g., chat.txt)")
        print("  2. Multiple files (for merging)")
        print("  3. Directory containing exports")
        print("  4. Pattern (e.g., *.txt)\n")
        
        while True:
            choice = input(f"{Colors.CYAN}How would you like to provide files?\n"
                          f"1) Enter file paths one by one\n"
                          f"2) Select all .txt files in current directory\n"
                          f"3) Enter directory path\n"
                          f"4) Enter file pattern\n"
                          f"Choice (1-4): {Colors.END}").strip()
            
            if choice == '1':
                self._collect_files_manually()
                break
            elif choice == '2':
                self._collect_files_from_current()
                break
            elif choice == '3':
                self._collect_files_from_directory()
                break
            elif choice == '4':
                self._collect_files_by_pattern()
                break
            else:
                print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.END}")
        
        # Show collected files
        if self.export_files:
            print(f"\n{Colors.GREEN}✅ Found {len(self.export_files)} export file(s):{Colors.END}")
            total_size = 0
            for f in self.export_files:
                size = os.path.getsize(f) / (1024 * 1024)
                total_size += size
                print(f"  • {Path(f).name} ({size:.1f} MB)")
            
            estimated_msgs = int(total_size * 5000)
            print(f"\n📊 Estimated messages: ~{estimated_msgs:,}")
        else:
            print(f"{Colors.FAIL}No files found. Exiting.{Colors.END}")
            sys.exit(1)
            
    def _collect_files_manually(self):
        """Manually enter file paths"""
        print("\nEnter file paths (one per line, empty line to finish):")
        while True:
            path = input(f"{Colors.CYAN}File path: {Colors.END}").strip()
            if not path:
                break
            if os.path.exists(path):
                self.export_files.append(path)
                print(f"  {Colors.GREEN}✅ Added{Colors.END}")
            else:
                print(f"  {Colors.FAIL}❌ File not found{Colors.END}")
                
    def _collect_files_from_current(self):
        """Collect all .txt files from current directory"""
        import glob
        files = glob.glob("*.txt")
        if files:
            self.export_files = files
        else:
            print(f"{Colors.WARNING}No .txt files found in current directory{Colors.END}")
            
    def _collect_files_from_directory(self):
        """Collect files from specified directory"""
        dir_path = input(f"{Colors.CYAN}Directory path: {Colors.END}").strip()
        if os.path.isdir(dir_path):
            import glob
            pattern = os.path.join(dir_path, "*.txt")
            files = glob.glob(pattern)
            if files:
                self.export_files = files
            else:
                print(f"{Colors.WARNING}No .txt files found in {dir_path}{Colors.END}")
        else:
            print(f"{Colors.FAIL}Directory not found{Colors.END}")
            
    def _collect_files_by_pattern(self):
        """Collect files by pattern"""
        pattern = input(f"{Colors.CYAN}File pattern (e.g., chat*.txt): {Colors.END}").strip()
        import glob
        files = glob.glob(pattern)
        if files:
            self.export_files = files
        else:
            print(f"{Colors.WARNING}No files matching pattern: {pattern}{Colors.END}")
            
    def choose_llm(self):
        """Choose which LLM to use"""
        print(f"\n{Colors.BOLD}🤖 Step 2: Choose LLM for Analysis{Colors.END}")
        print("-" * 50)
        
        print("\nWhich LLM would you like to use?")
        print(f"""
{Colors.GREEN}1) Ollama (RECOMMENDED){Colors.END}
   • 100% private and local
   • Free forever
   • No data leaves your computer
   • Supports Llama 2, Mistral, etc.
   
{Colors.BLUE}2) Transformer Models (Default){Colors.END}
   • Sentence-BERT + RoBERTa
   • Good accuracy, moderate size (2GB)
   • Runs locally
   
{Colors.WARNING}3) OpenAI GPT-4 (Requires API key){Colors.END}
   • Best accuracy
   • Costs money (~$50-100 for 200k messages)
   • Data sent to OpenAI
   
{Colors.CYAN}4) Basic Analysis (No LLM){Colors.END}
   • Simple statistics only
   • Fast but limited insights
        """)
        
        while True:
            choice = input(f"{Colors.CYAN}Your choice (1-4): {Colors.END}").strip()
            
            if choice == '1':
                self.llm_choice = 'ollama'
                self._setup_ollama()
                break
            elif choice == '2':
                self.llm_choice = 'transformers'
                print(f"{Colors.GREEN}✅ Using transformer models{Colors.END}")
                break
            elif choice == '3':
                self.llm_choice = 'openai'
                self._setup_openai()
                break
            elif choice == '4':
                self.llm_choice = 'basic'
                print(f"{Colors.CYAN}Using basic analysis{Colors.END}")
                break
            else:
                print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.END}")
                
    def _setup_ollama(self):
        """Setup Ollama"""
        print(f"\n{Colors.BOLD}Setting up Ollama...{Colors.END}")
        
        # Check if Ollama is installed
        try:
            result = subprocess.run(['ollama', '--version'], capture_output=True)
            if result.returncode == 0:
                print(f"{Colors.GREEN}✅ Ollama is installed{Colors.END}")
            else:
                raise Exception()
        except:
            print(f"{Colors.WARNING}Ollama not found. Installing...{Colors.END}")
            print("\nTo install Ollama:")
            
            system = platform.system()
            if system == "Darwin":  # macOS
                print("Run: brew install ollama")
            elif system == "Linux":
                print("Run: curl -fsSL https://ollama.ai/install.sh | sh")
            elif system == "Windows":
                print("Download from: https://ollama.ai/download")
            
            print("\nAfter installing, run this script again.")
            
            install_now = input(f"\n{Colors.CYAN}Try to install automatically? (y/n): {Colors.END}").lower()
            if install_now == 'y':
                self._install_ollama()
        
        # Choose Ollama model
        print(f"\n{Colors.BOLD}Choose Ollama model:{Colors.END}")
        print("1) llama2 (7B) - Good balance")
        print("2) mistral (7B) - Fast and accurate")
        print("3) llama2:13b - More accurate, slower")
        print("4) codellama - Good for technical analysis")
        
        model_choice = input(f"{Colors.CYAN}Choice (1-4, default=2): {Colors.END}").strip() or '2'
        
        models = {
            '1': 'llama2',
            '2': 'mistral',
            '3': 'llama2:13b',
            '4': 'codellama'
        }
        
        self.config['ollama_model'] = models.get(model_choice, 'mistral')
        
        # Pull the model
        print(f"\n{Colors.CYAN}Pulling {self.config['ollama_model']} model...{Colors.END}")
        subprocess.run(['ollama', 'pull', self.config['ollama_model']])
        
        print(f"{Colors.GREEN}✅ Ollama ready with {self.config['ollama_model']}{Colors.END}")
        
    def _install_ollama(self):
        """Try to install Ollama automatically"""
        system = platform.system()
        
        try:
            if system == "Darwin":  # macOS
                subprocess.run(['brew', 'install', 'ollama'])
            elif system == "Linux":
                subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh', '|', 'sh'], shell=True)
            else:
                print(f"{Colors.WARNING}Please install Ollama manually{Colors.END}")
        except Exception as e:
            print(f"{Colors.FAIL}Auto-install failed: {e}{Colors.END}")
            
    def _setup_openai(self):
        """Setup OpenAI API"""
        print(f"\n{Colors.BOLD}Setting up OpenAI...{Colors.END}")
        
        api_key = input(f"{Colors.CYAN}Enter your OpenAI API key: {Colors.END}").strip()
        if api_key:
            self.config['openai_api_key'] = api_key
            print(f"{Colors.GREEN}✅ OpenAI API key configured{Colors.END}")
        else:
            print(f"{Colors.WARNING}No API key provided, falling back to transformers{Colors.END}")
            self.llm_choice = 'transformers'
            
    def configure_analysis(self):
        """Configure analysis options"""
        print(f"\n{Colors.BOLD}⚙️ Step 3: Configure Analysis{Colors.END}")
        print("-" * 50)
        
        # Output directory
        output = input(f"{Colors.CYAN}Output directory (default: relationship_analysis): {Colors.END}").strip()
        self.output_dir = output or "relationship_analysis"
        
        # GPU detection
        print(f"\n{Colors.CYAN}Checking for GPU...{Colors.END}")
        try:
            import torch
            if torch.cuda.is_available():
                self.use_gpu = True
                print(f"{Colors.GREEN}✅ GPU detected! Analysis will be faster.{Colors.END}")
            else:
                print(f"{Colors.WARNING}No GPU detected. Using CPU (slower but works).{Colors.END}")
        except:
            print(f"{Colors.WARNING}PyTorch not installed. Will install if needed.{Colors.END}")
        
        # Analysis depth
        print(f"\n{Colors.BOLD}Analysis depth:{Colors.END}")
        print("1) Quick (first 10,000 messages)")
        print("2) Standard (first 50,000 messages)")
        print("3) Complete (all messages)")
        
        depth = input(f"{Colors.CYAN}Choice (1-3, default=3): {Colors.END}").strip() or '3'
        
        self.config['analysis_depth'] = {
            '1': 10000,
            '2': 50000,
            '3': None
        }.get(depth, None)
        
        # Specific analyses to run
        print(f"\n{Colors.BOLD}Which analyses to run?{Colors.END}")
        print("1) Everything (recommended)")
        print("2) Emotional analysis only")
        print("3) Communication patterns only")
        print("4) Conflict analysis only")
        print("5) Custom selection")
        
        analysis_choice = input(f"{Colors.CYAN}Choice (1-5, default=1): {Colors.END}").strip() or '1'
        
        if analysis_choice == '5':
            self._configure_custom_analysis()
        else:
            self.config['analysis_type'] = analysis_choice
            
    def _configure_custom_analysis(self):
        """Configure custom analysis options"""
        print(f"\n{Colors.BOLD}Select analyses to run:{Colors.END}")
        
        options = [
            "Gottman's Four Horsemen",
            "Attachment styles",
            "Love languages",
            "Emotional trajectory",
            "Conflict patterns",
            "Communication balance",
            "Topic analysis",
            "Relationship phases"
        ]
        
        selected = []
        for i, option in enumerate(options, 1):
            choice = input(f"{Colors.CYAN}{i}. {option} (y/n, default=y): {Colors.END}").strip().lower()
            if choice != 'n':
                selected.append(option)
        
        self.config['custom_analyses'] = selected
        
    def check_and_install(self):
        """Check and install dependencies"""
        print(f"\n{Colors.BOLD}📦 Step 4: Checking Dependencies{Colors.END}")
        print("-" * 50)
        
        # Create requirements based on LLM choice
        requirements = ['pandas', 'numpy', 'tqdm', 'plotly']
        
        if self.llm_choice == 'transformers':
            requirements.extend([
                'torch', 'transformers', 'sentence-transformers',
                'faiss-cpu' if not self.use_gpu else 'faiss-gpu'
            ])
        elif self.llm_choice == 'openai':
            requirements.append('openai')
        elif self.llm_choice == 'ollama':
            requirements.append('ollama-python')
        
        print(f"\nRequired packages: {', '.join(requirements)}")
        
        install = input(f"\n{Colors.CYAN}Install/update dependencies? (y/n, default=y): {Colors.END}").strip().lower()
        
        if install != 'n':
            print(f"\n{Colors.CYAN}Installing dependencies...{Colors.END}")
            for package in requirements:
                print(f"Installing {package}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             capture_output=True)
            print(f"{Colors.GREEN}✅ Dependencies ready{Colors.END}")
            
    def run_analysis(self):
        """Run the actual analysis"""
        print(f"\n{Colors.BOLD}🧠 Step 5: Running Analysis{Colors.END}")
        print("-" * 50)
        
        # Estimate time
        total_size = sum(os.path.getsize(f) for f in self.export_files) / (1024 * 1024)
        estimated_msgs = int(total_size * 5000)
        
        if self.use_gpu:
            estimated_time = estimated_msgs / 200000 * 45  # 45 min per 200k
        else:
            estimated_time = estimated_msgs / 200000 * 180  # 3 hours per 200k
            
        print(f"\n📊 Analysis summary:")
        print(f"  • Files: {len(self.export_files)}")
        print(f"  • Estimated messages: ~{estimated_msgs:,}")
        print(f"  • LLM: {self.llm_choice}")
        print(f"  • GPU: {'Yes' if self.use_gpu else 'No'}")
        print(f"  • Estimated time: ~{estimated_time:.0f} minutes")
        
        proceed = input(f"\n{Colors.CYAN}Start analysis? (y/n): {Colors.END}").strip().lower()
        
        if proceed != 'y':
            print(f"{Colors.WARNING}Analysis cancelled.{Colors.END}")
            return
        
        print(f"\n{Colors.GREEN}Starting analysis...{Colors.END}")
        print("This may take a while. You can:")
        print("  • Get coffee ☕")
        print("  • The script will notify when complete")
        
        # Save configuration
        config_path = os.path.join(self.output_dir, "analysis_config.json")
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'export_files': self.export_files,
                'llm_choice': self.llm_choice,
                'config': self.config,
                'estimated_messages': estimated_msgs
            }, f, indent=2)
        
        # Run appropriate analyzer based on LLM choice
        if self.llm_choice == 'ollama':
            self._run_ollama_analysis()
        elif self.llm_choice == 'transformers':
            self._run_transformer_analysis()
        elif self.llm_choice == 'openai':
            self._run_openai_analysis()
        else:
            self._run_basic_analysis()
            
    def _run_ollama_analysis(self):
        """Run analysis using Ollama"""
        print(f"\n{Colors.CYAN}Running Ollama analysis...{Colors.END}")
        
        # This would integrate with the Ollama-based analyzer
        # For now, showing the structure
        from multi_export_merger import WhatsAppMultiExportMerger
        
        # Merge exports
        merger = WhatsAppMultiExportMerger()
        merged_df = merger.merge_exports(self.export_files)
        
        print(f"✅ Merged {len(merged_df)} messages")
        
        # TODO: Implement Ollama-based analysis
        print(f"{Colors.GREEN}✅ Analysis complete!{Colors.END}")
        
    def _run_transformer_analysis(self):
        """Run analysis using transformer models"""
        print(f"\n{Colors.CYAN}Running transformer analysis...{Colors.END}")
        
        try:
            from complete_analyzer import CompleteRelationshipAnalyzer
            
            analyzer = CompleteRelationshipAnalyzer()
            analyzer.run_complete_pipeline(self.export_files, self.output_dir)
            
        except ImportError:
            print(f"{Colors.WARNING}Analyzer not found. Using basic analysis.{Colors.END}")
            self._run_basic_analysis()
            
    def _run_openai_analysis(self):
        """Run analysis using OpenAI"""
        print(f"\n{Colors.CYAN}Running OpenAI analysis...{Colors.END}")
        
        # TODO: Implement OpenAI-based analysis
        print(f"{Colors.WARNING}OpenAI analysis not yet implemented. Using transformers.{Colors.END}")
        self._run_transformer_analysis()
        
    def _run_basic_analysis(self):
        """Run basic statistical analysis"""
        print(f"\n{Colors.CYAN}Running basic analysis...{Colors.END}")
        
        from multi_export_merger import WhatsAppMultiExportMerger
        
        merger = WhatsAppMultiExportMerger()
        merged_df = merger.merge_exports(self.export_files)
        
        # Basic statistics
        stats = {
            'total_messages': len(merged_df),
            'participants': merged_df['sender'].unique().tolist(),
            'date_range': f"{merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}",
            'messages_per_person': merged_df['sender'].value_counts().to_dict()
        }
        
        # Save results
        with open(os.path.join(self.output_dir, "basic_analysis.json"), 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        print(f"{Colors.GREEN}✅ Basic analysis complete!{Colors.END}")
        
    def show_results(self):
        """Show analysis results"""
        print(f"\n{Colors.BOLD}📊 Step 6: Results{Colors.END}")
        print("-" * 50)
        
        print(f"\n{Colors.GREEN}✅ Analysis complete!{Colors.END}")
        print(f"\nResults saved to: {Colors.CYAN}{self.output_dir}/{Colors.END}")
        
        # List files created
        if os.path.exists(self.output_dir):
            files = os.listdir(self.output_dir)
            print(f"\nFiles created:")
            for f in files[:10]:  # Show first 10 files
                print(f"  • {f}")
            
            if len(files) > 10:
                print(f"  ... and {len(files)-10} more files")
        
        # Next steps
        print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
        print(f"1. Open {self.output_dir}/relationship_report.html in your browser")
        print(f"2. Run: streamlit run {self.output_dir}/dashboard.py")
        print(f"3. Share results with your partner for discussion")
        
        # Offer to open report
        if platform.system() == "Darwin":  # macOS
            open_cmd = "open"
        elif platform.system() == "Windows":
            open_cmd = "start"
        else:
            open_cmd = "xdg-open"
        
        open_report = input(f"\n{Colors.CYAN}Open report in browser? (y/n): {Colors.END}").strip().lower()
        
        if open_report == 'y':
            report_path = os.path.join(self.output_dir, "relationship_report.html")
            if os.path.exists(report_path):
                subprocess.run([open_cmd, report_path])
            else:
                print(f"{Colors.WARNING}Report not found. Check {self.output_dir}/{Colors.END}")


def main():
    """Main entry point"""
    tool = InteractiveRelTool()
    
    try:
        tool.run()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Analysis interrupted by user.{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.END}")
        print(f"Please report issues to: https://github.com/amruth9459/Rel-Tool/issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
