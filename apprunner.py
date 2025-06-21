import sys
import os
import subprocess
from threading import Thread
import time


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
        os.system("streamlit run client/ui.py")

    @staticmethod
    def run_analysis():
        """Run the data analysis pipeline (implementation not shown)"""
        print("Running data analysis pipeline...")
        os.system("python main.py")
        
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
        
   
        ApplicationRunner.run_analysis()
    
        api_thread = Thread(target=ApplicationRunner.run_flask_app)
        api_thread.daemon = True
        api_thread.start()
        
    
        time.sleep(3)
        

        ApplicationRunner.run_streamlit_ui()
    else:
        print("Available commands:")
        print("  --analysis : Run data analysis pipeline")
        print("  --api      : Start Flask API backend")
        print("  --ui       : Start Streamlit UI frontend")
        print("  --full     : Run analysis, then start API and UI")

if __name__ == '__main__':
    main()