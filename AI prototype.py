import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from io import BytesIO
import matplotlib.backends.backend_pdf as pdf_backend
import warnings

# Data Ingestion Class
class DataIngestion:
    def __init__(self, file_path, file_type):
        self.file_path = file_path
        self.file_type = file_type.lower()
        self.df = self.load_data()

    def load_data(self):
        try:
            if self.file_type == 'csv':
                df = pd.read_csv(self.file_path)
            elif self.file_type == 'json':
                df = pd.read_json(self.file_path)
            elif self.file_type == 'excel':
                df = pd.read_excel(self.file_path)
            else:
                raise ValueError(f"Unsupported file type: {self.file_type}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

# Data Processor Class
class DataProcessor:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        # Remove duplicates
        self.df.drop_duplicates(inplace=True)

        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())  # Impute with mean

        # Handle outliers
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df = self.df[(np.abs(stats.zscore(self.df[col])) < 3)]  # Z-score method
        self.df.reset_index(drop=True, inplace=True)

        # Validate correct data types
        self.validate_data_types()
        return self.df

    def validate_data_types(self):
        correct_dtypes = {
            'Rank': 'int',
            'Country': 'str',
            'Country Code': 'str',
            'Gold': 'int',
            'Silver': 'int',
            'Bronze': 'int',
            'Total': 'int'
        }

        for column, dtype in correct_dtypes.items():
            if column in self.df.columns:
                if dtype == 'int':
                    self.df[column] = pd.to_numeric(self.df[column], errors='coerce').fillna(0).astype(int)
                elif dtype == 'str':
                    self.df[column] = self.df[column].astype(str).fillna('Unknown')

    def preprocess_data(self):
        # Convert categorical columns to numerical
        categorical_cols = self.df.select_dtypes(include=[object]).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])

        # Standardize numerical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])

        return self.df

# Analysis Engine Class
class AnalysisEngine:
    def __init__(self, df):
        self.df = df

    def generate_statistics(self):
        return self.df.describe()

    def plot_medal_counts_distribution(self):
        self.df[['Gold', 'Silver', 'Bronze', 'Total']].hist(bins=10, figsize=(10, 7))
        plt.suptitle('Distribution of Medal Counts')
        plt.xlabel('Number of Medals')
        plt.ylabel('Frequency')
        plt.savefig('medal_counts_distribution.png')
        plt.close()

    def plot_total_medals_by_country(self):
        sns.barplot(x='Country', y='Total', data=self.df)
        plt.xticks(rotation=45)
        plt.title('Total Medals by Country')
        plt.xlabel('Country')
        plt.ylabel('Total Medals')
        plt.savefig('total_medals_by_country.png')
        plt.close()

    def plot_medal_breakdown(self):
        self.df.set_index('Country')[['Gold', 'Silver', 'Bronze']].plot(kind='bar', stacked=True, figsize=(10, 7))
        plt.title('Medal Breakdown by Country')
        plt.xlabel('Country')
        plt.ylabel('Number of Medals')
        plt.xticks(rotation=45)
        plt.savefig('medal_breakdown.png')
        plt.close()

    def plot_correlation_matrix(self):
        correlation_matrix = self.df[['Gold', 'Silver', 'Bronze', 'Total']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig('correlation_matrix.png')
        plt.close()

    def plot_rank_vs_total_medals(self):
        sns.scatterplot(x='Rank', y='Total', data=self.df)
        plt.title('Rank vs. Total Medals')
        plt.xlabel('Rank')
        plt.ylabel('Total Medals')
        plt.savefig('rank_vs_total_medals.png')
        plt.close()

    def regression(self):
        X = self.df[['Gold', 'Silver', 'Bronze']]
        y = self.df['Total']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model.coef_, model.intercept_, mse, r2

    def clustering(self, n_clusters=3):
        X = self.df[['Gold', 'Silver', 'Bronze']]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, self.df['Cluster'])
        return self.df[['Country', 'Cluster']], silhouette_avg

    def decision_tree_regression(self):
        X = self.df[['Gold', 'Silver', 'Bronze']]
        y = self.df['Total']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, mse, r2

    def generate_report(self):
        pdf_pages = pdf_backend.PdfPages("analysis_report.pdf")

        # Generate and save summary statistics plot
        stats = self.generate_statistics()
        plt.figure(figsize=(10, 7))
        stats.plot(kind='bar')
        plt.title('Summary Statistics')
        plt.xlabel('Features')
        plt.ylabel('Values')
        pdf_pages.savefig(bbox_inches='tight')
        plt.close()

        # Define plot methods and their filenames
        plot_methods = [
            (self.plot_medal_counts_distribution, 'medal_counts_distribution.png'),
            (self.plot_total_medals_by_country, 'total_medals_by_country.png'),
            (self.plot_medal_breakdown, 'medal_breakdown.png'),
            (self.plot_correlation_matrix, 'correlation_matrix.png'),
            (self.plot_rank_vs_total_medals, 'rank_vs_total_medals.png')
        ]

        for plot_method,_ in plot_methods:
            plot_method()
            pdf_pages.savefig(bbox_inches='tight')
            plt.close()

        # Linear regression results
        coef, intercept, mse, r2 = self.regression()
        plt.figure(figsize=(10, 7))
        plt.bar(range(len(coef)), coef)
        plt.title('Linear Regression Coefficients')
        plt.xlabel('Medal Type')
        plt.ylabel('Coefficient')
        pdf_pages.savefig(bbox_inches='tight')
        plt.close()

        with open("regression_results.txt", "w") as f:
            f.write(f"Linear Regression Coefficients: {coef}\n")
            f.write(f"Intercept: {intercept}\n")
            f.write(f"Mean Squared Error: {mse}\n")
            f.write(f"R-squared: {r2}\n")

        # Clustering results
        clusters, silhouette_avg = self.clustering()
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='Gold', y='Silver', hue='Cluster', data=self.df)
        plt.title(f'Clusters (Silhouette Score: {silhouette_avg:.2f})')
        pdf_pages.savefig(bbox_inches='tight')
        plt.close()

        # Decision tree results
        dt_model, dt_mse, dt_r2 = self.decision_tree_regression()
        with open("decision_tree_results.txt", "w") as f:
            f.write(f"Decision Tree MSE: {dt_mse}\n")
            f.write(f"Decision Tree R-squared: {dt_r2}\n")

        pdf_pages.close()
        print("Report generated successfully!")

# CLI Function
def run_cli(engine):
    print("Welcome to the Data Analysis CLI!")
    print("Available commands:")
    print("1. statistics - Generate summary statistics")
    print("2. plot_distribution - Plot distribution of medal counts")
    print("3. plot_medals_by_country - Plot total medals by country")
    print("4. plot_medal_breakdown - Plot medal breakdown")
    print("5. plot_correlation_matrix - Plot correlation matrix")
    print("6. plot_rank_vs_total_medals - Plot rank vs. total medals")
    print("7. regression - Perform linear regression")
    print("8. clustering - Perform K-Means clustering")
    print("9. decision_tree - Perform Decision Tree regression")
    print("10. report - Generate comprehensive report")
    print("11. exit - Exit the program")

    while True:
        command = input("\nEnter command: ").strip().lower()

        if command == "statistics":
            print(engine.generate_statistics())
        elif command == "plot_distribution":
            engine.plot_medal_counts_distribution()
            print("Plot saved as 'medal_counts_distribution.png'")
        elif command == "plot_medals_by_country":
            engine.plot_total_medals_by_country()
            print("Plot saved as 'total_medals_by_country.png'")
        elif command == "plot_medal_breakdown":
            engine.plot_medal_breakdown()
            print("Plot saved as 'medal_breakdown.png'")
        elif command == "plot_correlation_matrix":
            engine.plot_correlation_matrix()
            print("Plot saved as 'correlation_matrix.png'")
        elif command == "plot_rank_vs_total_medals":
            engine.plot_rank_vs_total_medals()
            print("Plot saved as 'rank_vs_total_medals.png'")
        elif command == "regression":
            coef, intercept, mse, r2 = engine.regression()
            print(f"Coefficients: {coef}")
            print(f"Intercept: {intercept}")
            print(f"Mean Squared Error: {mse}")
            print(f"R-squared: {r2}")
        elif command == "clustering":
            clusters, silhouette_avg = engine.clustering()
            print(clusters)
            print(f"Silhouette Score: {silhouette_avg}")
        elif command == "decision_tree":
            model, mse, r2 = engine.decision_tree_regression()
            print(f"Decision Tree MSE: {mse}")
            print(f"Decision Tree R-squared: {r2}")
        elif command == "report":
            engine.generate_report()
        elif command == "exit":
            break
        else:
            print("Unknown command. Please try again.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=Warning)

    # Load the data
    df = pd.read_excel('C:/Users/kalya/Documents/PycharmProjects/FeatureEngineering/olympics2024.xlsx')

    # Initialize Data Processor and clean/preprocess data
    processor = DataProcessor(df)
    df_cleaned = processor.clean_data()
    df_preprocessed = processor.preprocess_data()

    # Initialize Analysis Engine
    engine = AnalysisEngine(df_preprocessed)

    # Run CLI
    run_cli(engine)
