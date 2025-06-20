import sys
import os
import subprocess
from threading import Thread
import time
import main

class ApplicationRunner:
    @staticmethod
    def run_flask_app():
        """Run the Flask API backend"""
        print("Starting Flask API server...")
        os.system("python application.py")

    @staticmethod
    def run_streamlit_ui():
        """Run the Streamlit UI frontend"""
        print("Starting Streamlit UI...")
        os.system("streamlit run ui.py")

    @staticmethod
    def run_analysis():
        """Run the data analysis pipeline (implementation not shown)"""
        print("Running data analysis pipeline...")
        os.system("python main.py")
        # Your analysis code here
        print("Analysis complete! Outputs saved in 'output/' directory")

def main():
    if '--api' in sys.argv:
        ApplicationRunner.run_flask_app()
    elif '--ui' in sys.argv:
        ApplicationRunner.run_streamlit_ui()
    elif '--analysis' in sys.argv:
        ApplicationRunner.run_analysis()
    elif '--full' in sys.argv:
        print("Running full workflow: Analysis -> API -> UI")
        
        # Run analysis first
        ApplicationRunner.run_analysis()
        
        # Start API in background thread
        api_thread = Thread(target=ApplicationRunner.run_flask_app)
        api_thread.daemon = True
        api_thread.start()
        
        # Wait for API to start
        time.sleep(3)
        
        # Start UI
        ApplicationRunner.run_streamlit_ui()
    else:
        print("Available commands:")
        print("  --analysis : Run data analysis pipeline")
        print("  --api      : Start Flask API backend")
        print("  --ui       : Start Streamlit UI frontend")
        print("  --full     : Run analysis, then start API and UI")

if __name__ == '__main__':
    main()