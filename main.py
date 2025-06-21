import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from server.cluster_app import Pipeline

if __name__=='__main__':
    pipemain=Pipeline()
    pipemain.main()