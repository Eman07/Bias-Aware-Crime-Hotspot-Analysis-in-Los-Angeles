import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, asin

class LACrimeAnalyzer:
    def __init__(self, start_date='2022-01-01', end_date='2023-12-31'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.crime_data = None
        self.station_data = None
        self.census_data = None
        
    def load_data(self, crime_file, station_file, census_file):
        """Load data from provided files"""
        # Load crime data
        self.crime_data = pd.read_csv('/Users/emmanuelephraim/Downloads/Crime_Data_from_2020_to_Present_20241215.csv')
        
        # Load police station locations
        self.station_data = pd.read_csv('/Users/emmanuelephraim/Downloads/LAPD_Police_Stations.csv')
        
        # Load census data
        self.census_data = pd.read_csv('/Users/emmanuelephraim/Downloads/ACSST1Y2023.S1903-2024-12-16T043844.csv')
        
    def process_data(self):
        """Clean and process all data sources"""
        # Process crime data
        self.crime_data['DATE'] = pd.to_datetime(
            self.crime_data['DATE OCC'],
            format='%m/%d/%Y %I:%M:%S %p'
        )
        
        # Filter date range
        mask = (self.crime_data['DATE'] >= self.start_date) & \
               (self.crime_data['DATE'] <= self.end_date)
        self.crime_data = self.crime_data[mask]
        
        # Process station data
        # Convert state plane coordinates to lat/lon
        
        
        # Process census data - extract median income by race
        income_data = self.census_data[
            self.census_data['Label (Grouping)'].str.contains('White alone, not Hispanic')
        ]['Los Angeles County, California!!Median income (dollars)!!Estimate'].iloc[0]
        
        # Add census data to crime data (using area as proxy since tract isn't available)
        self.crime_data['median_income'] = income_data
        
        # Calculate distances to nearest station
        self.crime_data['distance_to_station'] = self.calculate_station_distances()
        
        return self.crime_data
    
    def calculate_station_distances(self):
        """Calculate distance to nearest police station"""
        distances = []
        for _, crime in self.crime_data.iterrows():
            min_dist = float('inf')
            for _, station in self.station_data.iterrows():
                dist = self.haversine_distance(
                    float(crime['LAT']), float(crime['LON']),
                    float(station['LAT']), float(station['LON'])
                )
                min_dist = min(min_dist, dist)
            distances.append(min_dist)
        return distances
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return R * c
    
    def calculate_bias_metrics(self):
        """Calculate bias-related metrics"""
        # Calculate crime rates by area
        self.crime_data['crime_rate'] = self.crime_data.groupby('AREA')[
            'DR_NO'
        ].transform('count')
        
        # Calculate arrest rates (using Status as proxy)
        self.crime_data['arrested'] = self.crime_data['Status'].isin(['AA', 'AO'])
        self.crime_data['arrest_rate'] = self.crime_data.groupby('AREA')[
            'arrested'
        ].transform('mean')
        
        return self.crime_data
    
    def detect_hotspots(self, n_clusters=5):
        """Perform bias-aware clustering"""
        # Prepare features
        features = ['LAT', 'LON', 'crime_rate', 'distance_to_station']
        X = self.crime_data[features].copy()
        
        # Handle any missing values
        X = X.fillna(X.mean())
        X = StandardScaler().fit_transform(X)
        
        # Calculate weights
        weights = self.calculate_fairness_weights()
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.crime_data['cluster'] = kmeans.fit_predict(X, sample_weight=weights)
        
        return self.crime_data, kmeans.cluster_centers_
    
    def calculate_fairness_weights(self):
        """Calculate weights to adjust for bias"""
        weights = pd.DataFrame()
        
        # Distance-based weight
        weights['distance_weight'] = (
            self.crime_data['distance_to_station'] / 
            self.crime_data['distance_to_station'].max()
        )
        
        # Crime rate weight
        weights['crime_weight'] = (
            1 / self.crime_data['crime_rate']
        )
        
        return weights.mean(axis=1)
    
    def analyze_clusters(self):
        """Analyze cluster characteristics"""
        cluster_stats = []
        
        for cluster in self.crime_data['cluster'].unique():
            cluster_data = self.crime_data[
                self.crime_data['cluster'] == cluster
            ]
            
            stats = {
                'cluster': cluster,
                'size': len(cluster_data),
                'avg_distance': cluster_data['distance_to_station'].mean(),
                'crime_types': cluster_data['Crm Cd Desc'].value_counts().head(3),
                'avg_arrest_rate': cluster_data['arrest_rate'].mean()
            }
            cluster_stats.append(stats)
            
        return pd.DataFrame(cluster_stats)
    
    def visualize_results(self):
        """Create visualizations"""
        # Set up figure with two subplots side by side
        fig = plt.figure(figsize=(20, 8))
        
        # First subplot: Crime Hotspots Map
        ax1 = fig.add_subplot(121)
        
        # LA County approximate boundaries
        la_bounds = {
            'lat_min': 33.7037,
            'lat_max': 34.3373,
            'lon_min': -118.6682,
            'lon_max': -118.1553
        }
        
        # Create scatter plot with bounded data
        scatter = ax1.scatter(
            self.crime_data['LON'],
            self.crime_data['LAT'],
            c=self.crime_data['cluster'],
            cmap='viridis',
            alpha=0.4,
            s=20
        )
        
        # Add police stations to the plot
        ax1.scatter(
            self.station_data['LON'],
            self.station_data['LAT'],
            c='red',
            marker='^',
            s=100,
            label='Police Stations'
        )
        
        # Set plot boundaries and customize
        ax1.set_xlim(la_bounds['lon_min'], la_bounds['lon_max'])
        ax1.set_ylim(la_bounds['lat_min'], la_bounds['lat_max'])
        ax1.set_title('Crime Hotspots in Los Angeles (2020)')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax1, label='Cluster')
        
        # Second subplot: Arrest Rate Distribution
        ax2 = fig.add_subplot(122)
        sns.boxplot(
            x='cluster',
            y='arrest_rate',
            data=self.crime_data,
            ax=ax2
        )
        ax2.set_title('Arrest Rate Distribution by Cluster')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Arrest Rate')
        
        # Add cluster size annotations
        cluster_sizes = self.crime_data['cluster'].value_counts()
        for i, size in cluster_sizes.items():
            ax2.text(i, ax2.get_ylim()[0], f'n={size}', 
                    horizontalalignment='center', verticalalignment='top')
        
        # Adjust layout
        plt.tight_layout()
        plt.show()

def main():
    # Initialize analyzer
    analyzer = LACrimeAnalyzer()
    
    # Load and process data
    print("Loading data...")
    analyzer.load_data(
        'crime_data.csv',
        'LAPD_Police_Stations.csv',
        'census_data.csv'
    )
    
    print("Processing data...")
    analyzer.process_data()
    
    # Calculate bias metrics
    print("Calculating bias metrics...")
    analyzer.calculate_bias_metrics()
    
    # Detect hotspots
    print("Detecting hotspots...")
    data, centers = analyzer.detect_hotspots()
    
    # Analyze results
    print("Analyzing clusters...")
    cluster_analysis = analyzer.analyze_clusters()
    print("\nCluster Analysis:")
    print(cluster_analysis)
    
    # Visualize results
    print("Creating visualizations...")
    analyzer.visualize_results()

if __name__ == "__main__":
    main()
