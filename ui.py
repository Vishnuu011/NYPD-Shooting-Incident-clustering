import streamlit as st
import folium
from streamlit_folium import folium_static
import requests
import os
import json

# Set page config
st.set_page_config(page_title="NYPD Shooting Analysis", layout="wide")

# API endpoint configuration
API_URL = "http://localhost:5000/predict"

def main():
    st.title("NYPD Shooting Incident Cluster Prediction")
    st.markdown("""
    This tool predicts crime clusters based on historical NYPD shooting data.
    Enter coordinates to see which cluster the location belongs to.
    """)
    
    # Load cluster map if available
    cluster_map_exists = os.path.exists('output/cluster_map.html')
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Predict New Location")
        lat = st.number_input("Latitude", value=40.7128, format="%.6f", step=0.0001)
        lon = st.number_input("Longitude", value=-74.0060, format="%.6f", step=0.0001)
        
        if st.button("Predict Cluster", help="Predict cluster for the entered coordinates"):
            try:
                # Call backend API
                response = requests.post(
                    API_URL,
                    json={'latitude': lat, 'longitude': lon},
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    cluster = result['cluster']
                    prob = result['probability']
                    
                    if cluster == -1:
                        st.warning("This location doesn't belong to any major crime cluster")
                    else:
                        st.success(f"Predicted Cluster: **{cluster}** (Probability: {prob:.2%})")
                    
                    # Show on map
                    st.subheader("Location on Map")
                    m = folium.Map(location=[lat, lon], zoom_start=14)
                    folium.Marker(
                        [lat, lon], 
                        popup=f"Predicted Cluster: {cluster}",
                        icon=folium.Icon(color='red' if cluster == -1 else 'blue')
                    ).add_to(m)
                    folium_static(m)
                else:
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the prediction API. Please make sure the backend is running.")
            except requests.exceptions.Timeout:
                st.error("API request timed out. Please try again.")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    with col2:
        st.subheader("Cluster Map")
        if cluster_map_exists:
            with open('output/cluster_map.html', 'r') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600)
        else:
            st.warning("Cluster map not available. Run the analysis first.")
    
    st.divider()
    st.subheader("Cluster Information")
    st.markdown("""
    - **Cluster -1**: Noise points (not part of any cluster)
    - **Other clusters**: Grouped crime hotspots identified by the algorithm
    - **Probability**: Confidence level of the cluster assignment
    """)
    
    st.subheader("Analysis Insights")
    if os.path.exists('output/analysis_insights.txt'):
        with open('output/analysis_insights.txt', 'r') as f:
            insights = f.read()
        st.text(insights)
    else:
        st.warning("No insights available. Run the analysis first.")

if __name__ == '__main__':
    main()