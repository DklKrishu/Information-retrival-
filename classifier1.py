import os
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
import threading
from sklearn.calibration import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

NEWS_DATA_PATH = os.path.join(os.path.dirname(__file__), 'news_classification_dataset.csv')
NEWS_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'news_classifier_model.joblib')

def retrieve_saved_model():
    if not os.path.exists(NEWS_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {NEWS_MODEL_PATH}. Please run classifier1.py to train and create it.")
    
    saved_data = joblib.load(NEWS_MODEL_PATH)
    trained_model = saved_data['model']
    category_labels = saved_data['labels']
    return trained_model, category_labels

def extract_news_data(csv_path: str) -> tuple[list, list, list]:
    try:
        news_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{csv_path}'.")
        print("Please make sure the CSV file is in the same directory.")
        return list(), list(), list()

    print("--- News Data Loaded Successfully ---")
    print("DataFrame Info:")
    news_df.info()
    print("\nFirst 5 rows of the DataFrame:")
    print(news_df.head())
    print("-" * 30)

    news_df.dropna(subset=['text', 'category'], inplace=True)
    
    document_texts = news_df['text'].tolist()
    
    category_codes, unique_categories = pd.factorize(news_df['category'])
    
    print(f"Loaded {len(document_texts)} news articles.")
    print(f"Found categories: {list(unique_categories)}")
    
    return document_texts, category_codes.tolist(), list(unique_categories)

def display_confusion_matrix(confusion_mat, category_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=category_names, yticklabels=category_names, ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    
    plt.show()

def build_and_assess_model(document_texts: list, category_codes: list, unique_categories: list):
    news_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=2)),
        ('clf', MultinomialNB(alpha=0.1))
    ])

    print("\n--- Evaluating model with K-Fold Cross-Validation ---")

    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    predicted_labels = cross_val_predict(news_pipeline, np.array(document_texts), category_codes, cv=cv_strategy)

    model_accuracy = accuracy_score(category_codes, predicted_labels)
    macro_f1_score = f1_score(category_codes, predicted_labels, average='macro') 
    
    print(f"Cross-Validated Accuracy: {model_accuracy:.4f}")
    print(f"Cross-Validated Macro F1-Score: {macro_f1_score:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(category_codes, predicted_labels, target_names=unique_categories))
    
    conf_matrix = confusion_matrix(category_codes, predicted_labels)
    display_confusion_matrix(conf_matrix, unique_categories)
    
    print("\nRetraining the model on the full dataset for production...")
    production_model = news_pipeline.fit(document_texts, category_codes)
    
    return production_model

def execute_news_classifier():
    news_model = None
    category_names = []

    try:
        print(f"Attempting to load pre-trained model from {NEWS_MODEL_PATH}...")
        news_model, category_names = retrieve_saved_model()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No pre-trained model found. Training a new one from CSV.")
        document_texts, category_codes, categories_from_data = extract_news_data(NEWS_DATA_PATH)
        
        if not document_texts:
            print("\nExiting: Could not load data.")
            return
            
        category_names = categories_from_data
        
        news_model = build_and_assess_model(document_texts, category_codes, category_names)
        
        print(f"\nSaving model to {NEWS_MODEL_PATH}...")
        os.makedirs(os.path.dirname(NEWS_MODEL_PATH), exist_ok=True)
        joblib.dump({'model': news_model, 'labels': category_names}, NEWS_MODEL_PATH)
        print("Model saved.")

    print("\n--- News Document Classifier Ready ---")
    print("Enter a sentence or a paragraph to classify.")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        text_input = input("\nEnter text> ")
        if text_input.lower() in ['exit', 'quit']:
            break
        
        if not text_input.strip():
            continue

        predicted_category_index = news_model.predict([text_input])[0]
        predicted_category = category_names[predicted_category_index]
        
        prediction_probabilities = news_model.predict_proba([text_input])[0]
        prediction_confidence = prediction_probabilities[predicted_category_index]
        
        print(f"\n=> Predicted Category: ** {predicted_category.upper()} **")
        print(f"   Confidence: {prediction_confidence:.2%}")

class NewsClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("News Document Classification System")
        self.root.geometry("1200x700")  # Reduced size for lower resolutions
        self.root.configure(bg='#2c3e50')  # Dark background
        
        # Make window resizable and set minimum size
        self.root.minsize(1000, 600)
        
        # Configure proper window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure styles
        self.setup_styles()
        
        # Initialize variables
        self.model = None
        self.category_names = []
        self.training_data = None
        self.is_training = False
        self.is_evaluating = False
        
        # Create GUI components
        self.create_widgets()
        
        # Try to load existing model
        self.load_existing_model()

    def on_closing(self):
        """Handle application closing properly"""
        try:
            # Stop any running threads
            if self.is_training or self.is_evaluating:
                pass  # No progress bar to stop
            
            # Close matplotlib figures properly
            plt.close('all')
            
            # Destroy the window
            self.root.quit()
            self.root.destroy()
            
        except Exception:
            pass
        finally:
            # Force exit the application
            import sys
            sys.exit(0)
    
    def setup_styles(self):
        """Configure ttk styles for better appearance"""
        style = ttk.Style()
        style.theme_use('clam')  # Use clam theme for better dark mode support
        
        # Configure button style with smaller font for better fit
        style.configure('Large.TButton',
                       font=('Arial', 12, 'bold'),
                       padding=(15, 8),
                       background='#34495e',
                       foreground='white',
                       borderwidth=1,
                       focuscolor='none')
        
        style.map('Large.TButton',
                 background=[('active', '#5d6d7e'),
                           ('pressed', '#1b2631')])
        
        # Configure label style with smaller font
        style.configure('Large.TLabel',
                       font=('Arial', 12),
                       background='#2c3e50',
                       foreground='white')
        
        # Configure header style with smaller font
        style.configure('Header.TLabel',
                       font=('Arial', 18, 'bold'),
                       background='#2c3e50',
                       foreground='#3498db')
        
        # Configure frame style for dark theme
        style.configure('Card.TFrame',
                       background='#34495e',
                       relief='solid',
                       borderwidth=1)
        
        # Configure button frame style
        style.configure('ButtonFrame.TFrame',
                       background='#34495e')
        
        # Configure notebook style for dark theme
        style.configure('TNotebook',
                       background='#2c3e50',
                       tabposition='n')
        
        style.configure('TNotebook.Tab',
                       font=('Arial', 14, 'bold'),
                       padding=(20, 10),
                       background='#34495e',
                       foreground='white')
        
        style.map('TNotebook.Tab',
                 background=[('selected', '#3498db'),
                           ('active', '#5d6d7e')])
        
        # Configure LabelFrame for dark theme
        style.configure('TLabelframe',
                       background='#34495e',
                       foreground='white',
                       borderwidth=2)
        
        style.configure('TLabelframe.Label',
                       font=('Arial', 14, 'bold'),
                       background='#34495e',
                       foreground='white')
    
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Main title
        title_label = ttk.Label(self.root, text="News Document Classification System", 
                               style='Header.TLabel')
        title_label.pack(pady=15)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Training tab
        self.training_frame = ttk.Frame(notebook)
        self.training_frame.configure(style='Card.TFrame')
        notebook.add(self.training_frame, text="Training & Evaluation")
        self.create_training_tab()
        
        # Classification tab
        self.classification_frame = ttk.Frame(notebook)
        self.classification_frame.configure(style='Card.TFrame')
        notebook.add(self.classification_frame, text="Text Classification")
        self.create_classification_tab()
    
    def create_training_tab(self):
        """Create training and evaluation interface"""
        # Data section
        data_frame = ttk.LabelFrame(self.training_frame, text="Data Management", 
                                   padding=15)
        data_frame.pack(fill=tk.X, padx=15, pady=8)
        
        ttk.Button(data_frame, text="Load Training Data", 
                  command=self.load_training_data, 
                  style='Large.TButton').pack(side=tk.LEFT, padx=10)
        
        self.data_status_label = ttk.Label(data_frame, text="No data loaded", 
                                          style='Large.TLabel')
        self.data_status_label.pack(side=tk.LEFT, padx=15)
        
        # Training section
        training_frame = ttk.LabelFrame(self.training_frame, text="Model Training & Evaluation", 
                                       padding=15)
        training_frame.pack(fill=tk.X, padx=15, pady=8)
        
        # Buttons row
        button_row = ttk.Frame(training_frame, style='ButtonFrame.TFrame')
        button_row.pack(fill=tk.X, pady=8)
        
        ttk.Button(button_row, text="Train Model", 
                  command=self.train_model_thread, 
                  style='Large.TButton').pack(side=tk.LEFT, padx=8)
        
        ttk.Button(button_row, text="Evaluate Model", 
                  command=self.evaluate_model_thread, 
                  style='Large.TButton').pack(side=tk.LEFT, padx=8)
        
        ttk.Button(button_row, text="Show Confusion Matrix", 
                  command=self.show_confusion_matrix, 
                  style='Large.TButton').pack(side=tk.LEFT, padx=8)
        
        # Status and progress
        status_frame = ttk.Frame(training_frame)
        status_frame.pack(fill=tk.X, pady=8)
        
        self.training_status_label = ttk.Label(status_frame, text="Ready to train", 
                                              style='Large.TLabel')
        self.training_status_label.pack(side=tk.LEFT, padx=15)
        
        # Metrics display (full width)
        metrics_frame = ttk.LabelFrame(self.training_frame, text="Performance Metrics", 
                                      padding=15)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=8)
        
        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, height=18, 
                                                     font=('Courier', 10),
                                                     bg='#2c3e50', fg='white',
                                                     insertbackground='white')
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
    
    def create_classification_tab(self):
        """Create text classification interface"""
        # Input section
        input_frame = ttk.LabelFrame(self.classification_frame, text="Text Input", 
                                    padding=15)
        input_frame.pack(fill=tk.X, padx=15, pady=8)
        
        ttk.Label(input_frame, text="Enter text to classify:", 
                 style='Large.TLabel').pack(anchor=tk.W, pady=8)
        
        self.text_input = scrolledtext.ScrolledText(input_frame, height=6, 
                                                   font=('Arial', 14),
                                                   bg='#2c3e50', fg='white',
                                                   insertbackground='white')
        self.text_input.pack(fill=tk.X, pady=8)
        
        # Buttons
        button_frame = ttk.Frame(input_frame, style='ButtonFrame.TFrame')
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Classify Text", 
                  command=self.classify_text, 
                  style='Large.TButton').pack(side=tk.LEFT, padx=8)
        
        ttk.Button(button_frame, text="Clear Input", 
                  command=self.clear_input, 
                  style='Large.TButton').pack(side=tk.LEFT, padx=8)
        
        ttk.Button(button_frame, text="Show Confidence Scores", 
                  command=self.show_confidence_scores, 
                  style='Large.TButton').pack(side=tk.LEFT, padx=8)
        
        # Results section (full width)
        results_frame = ttk.LabelFrame(self.classification_frame, text="Classification Results", 
                                      padding=15)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=8)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, 
                                                     font=('Arial', 12),
                                                     bg='#2c3e50', fg='white',
                                                     insertbackground='white')
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def load_existing_model(self):
        """Try to load existing model"""
        try:
            self.model, self.category_names = retrieve_saved_model()
            self.training_status_label.config(text="Model loaded successfully", style='ButtonFrame.TFrame')
            self.update_metrics_display("Model loaded from file.\n")
        except FileNotFoundError:
            self.training_status_label.config(text="No existing model found")
    
    def load_training_data(self):
        """Load training data from CSV file"""
        try:
            document_texts, category_codes, categories = extract_news_data(NEWS_DATA_PATH)
            if document_texts:
                self.training_data = (document_texts, category_codes, categories)
                self.category_names = categories
                self.data_status_label.config(text=f"Loaded {len(document_texts)} documents")
                self.update_metrics_display(f"Training data loaded: {len(document_texts)} documents\n")
                self.update_metrics_display(f"Categories: {', '.join(categories)}\n\n")
            else:
                messagebox.showerror("Error", "Could not load training data")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def train_model_thread(self):
        """Train model in separate thread"""
        if not self.training_data:
            messagebox.showwarning("Warning", "Please load training data first")
            return
        
        if self.is_training:
            return
        
        self.is_training = True
        thread = threading.Thread(target=self.train_model_worker)
        thread.daemon = True
        thread.start()
    
    def train_model_worker(self):
        """Worker function for training model in background thread"""
        try:
            # Update UI on main thread
            self.root.after(0, lambda: self.training_status_label.config(text="Training model..."))
            
            if self.training_data is None:
                raise ValueError("No training data available. Please load training data first.")
            
            document_texts, category_codes, categories = self.training_data
            
            # Create and train model
            news_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=2)),
                ('clf', MultinomialNB(alpha=0.1))
            ])
            
            # Train the model
            trained_model = news_pipeline.fit(document_texts, category_codes)
            
            # Save model
            os.makedirs(os.path.dirname(NEWS_MODEL_PATH), exist_ok=True)
            joblib.dump({'model': trained_model, 'labels': categories}, NEWS_MODEL_PATH)
            
            # Update UI on main thread
            self.root.after(0, lambda: self.on_training_complete(trained_model, categories))
            
        except Exception as e:
            # Update UI on main thread
            error_msg = str(e)
            self.root.after(0, lambda: self.on_training_error(error_msg))
        finally:
            self.is_training = False
    
    def on_training_complete(self, trained_model, categories):
        """Called on main thread when training completes"""
        self.model = trained_model
        self.category_names = categories
        self.training_status_label.config(text="Model trained successfully")
        self.update_metrics_display("Model training completed and saved.\n")
    
    def on_training_error(self, error_msg):
        """Called on main thread when training fails"""
        self.training_status_label.config(text="Training failed")
        messagebox.showerror("Error", f"Training failed: {error_msg}")
    
    def evaluate_model_thread(self):
        """Evaluate model in separate thread"""
        if not self.model or not self.training_data:
            messagebox.showwarning("Warning", "Please load data and train model first")
            return
        
        if self.is_evaluating:
            return
        
        self.is_evaluating = True
        thread = threading.Thread(target=self.evaluate_model_worker)
        thread.daemon = True
        thread.start()
    
    def evaluate_model_worker(self):
        """Worker function for evaluating model in background thread"""
        try:
            # Update UI on main thread
            self.root.after(0, lambda: self.training_status_label.config(text="Evaluating model..."))
            
            if self.training_data is None:
                raise ValueError("No training data available. Please load training data first.")
            
            document_texts, category_codes, categories = self.training_data
            
            # Create a fresh pipeline for cross-validation
            from sklearn.model_selection import cross_val_score
            fresh_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=2)),
                ('clf', MultinomialNB(alpha=0.1))
            ])
            cv_scores = cross_val_score(fresh_pipeline, np.array(document_texts), category_codes, cv=5)
            
            # Get predictions for classification report
            predicted_labels = cross_val_predict(fresh_pipeline, np.array(document_texts), category_codes, cv=5)
            
            # Calculate metrics
            accuracy = accuracy_score(category_codes, predicted_labels)
            f1 = f1_score(category_codes, predicted_labels, average='macro')
            report = classification_report(category_codes, predicted_labels, target_names=categories)
            
            # Store confusion matrix data for visualization
            cm = confusion_matrix(category_codes, predicted_labels)
            
            # Update UI on main thread
            self.root.after(0, lambda: self.on_evaluation_complete(accuracy, f1, report, cv_scores, cm, categories))
            
        except Exception as e:
            # Update UI on main thread
            error_msg = str(e)
            self.root.after(0, lambda: self.on_evaluation_error(error_msg))
        finally:
            self.is_evaluating = False
    
    def on_evaluation_complete(self, accuracy, f1, report, cv_scores, cm, categories):
        """Called on main thread when evaluation completes"""
        self.training_status_label.config(text="Evaluation completed")
        
        # Store confusion matrix data for visualization
        self.last_confusion_matrix = cm
        self.last_categories = categories
        
        # Update metrics display
        metrics_text = "=== MODEL EVALUATION RESULTS ===\n\n"
        metrics_text += f"Cross-Validation Accuracy: {accuracy:.4f} (±{cv_scores.std()*2:.4f})\n"
        metrics_text += f"Macro F1-Score: {f1:.4f}\n\n"
        metrics_text += "Classification Report:\n"
        metrics_text += str(report) + "\n"
        
        self.update_metrics_display(metrics_text)
    
    def on_evaluation_error(self, error_msg):
        """Called on main thread when evaluation fails"""
        self.training_status_label.config(text="Evaluation failed")
        messagebox.showerror("Error", f"Evaluation failed: {error_msg}")
    
    def classify_text(self):
        """Classify the input text"""
        if not self.model:
            messagebox.showwarning("Warning", "Please train or load a model first")
            return
        
        text = self.text_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to classify")
            return
        
        try:
            # Get prediction
            predicted_category_index = self.model.predict([text])[0]
            predicted_category = self.category_names[predicted_category_index]
            
            # Get probabilities
            probabilities = self.model.predict_proba([text])[0]
            confidence = probabilities[predicted_category_index]
            
            # Display results
            result_text = "=== CLASSIFICATION RESULTS ===\n\n"
            result_text += f"Input Text: {text[:200]}{'...' if len(text) > 200 else ''}\n\n"
            result_text += f"Predicted Category: {predicted_category.upper()}\n"
            result_text += f"Confidence: {confidence:.2%}\n\n"
            result_text += "All Category Probabilities:\n"
            
            for i, (category, prob) in enumerate(zip(self.category_names, probabilities)):
                result_text += f"  {category}: {prob:.3f} ({prob*100:.1f}%)\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, result_text)
            
            # Store for visualization
            self.last_probabilities = probabilities
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
    
    def clear_input(self):
        """Clear the input text"""
        self.text_input.delete(1.0, tk.END)
        self.results_text.delete(1.0, tk.END)
    
    def show_confusion_matrix(self):
        """Display confusion matrix in a separate window"""
        if hasattr(self, 'last_confusion_matrix') and hasattr(self, 'last_categories'):
            # Use stored confusion matrix from evaluation
            cm = self.last_confusion_matrix
            categories = self.last_categories
        elif self.model and self.training_data:
            # Generate confusion matrix if not available
            try:
                document_texts, category_codes, categories = self.training_data
                # Create fresh pipeline to avoid threading issues
                fresh_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=2)),
                    ('clf', MultinomialNB(alpha=0.1))
                ])
                predicted_labels = cross_val_predict(fresh_pipeline, np.array(document_texts), category_codes, cv=5)
                cm = confusion_matrix(category_codes, predicted_labels)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create confusion matrix: {str(e)}")
                return
        else:
            messagebox.showwarning("Warning", "Please train and evaluate model first")
            return
        
        try:
            # Create a new window for confusion matrix
            import matplotlib
            matplotlib.use('TkAgg')
            
            # Close any existing matplotlib figures to prevent multiple windows
            plt.close('all')
            
            # Create new figure with white background for better text visibility
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            # Create a custom colormap that ensures good contrast for text
            import matplotlib.colors as mcolors
            # Use colors that provide good contrast for both light and dark text
            colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
            n_bins = 256
            cmap = mcolors.LinearSegmentedColormap.from_list('custom_blues', colors, N=n_bins)
            
            # Create the heatmap with dynamic text color based on cell value
            # Get the max value to determine text color thresholds
            max_val = cm.max()
            threshold = max_val / 2
            
            # Create heatmap
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            
            # Add text annotations with dynamic color
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    # Use white text for dark cells, black text for light cells
                    text_color = 'white' if cm[i, j] > threshold else 'black'
                    ax.text(j, i, format(cm[i, j], 'd'),
                                 ha="center", va="center", color=text_color,
                                 fontsize=14, fontweight='bold')
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(categories)))
            ax.set_yticks(np.arange(len(categories)))
            ax.set_xticklabels(categories)
            ax.set_yticklabels(categories)
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Set title and labels
            ax.set_title('Confusion Matrix', fontsize=20, fontweight='bold', 
                        pad=20, color='black')
            ax.set_ylabel('Actual Category', fontsize=16, color='black')
            ax.set_xlabel('Predicted Category', fontsize=16, color='black')
            ax.tick_params(labelsize=12, colors='black')
            
            # Add grid lines
            ax.set_xticks(np.arange(len(categories)+1)-.5, minor=True)
            ax.set_yticks(np.arange(len(categories)+1)-.5, minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
            ax.tick_params(which="minor", size=0)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display confusion matrix: {str(e)}")
    
    def show_confidence_scores(self):
        """Display confidence scores in a separate window"""
        if not hasattr(self, 'last_probabilities'):
            messagebox.showwarning("Warning", "Please classify some text first")
            return
        
        try:
            # Create a new window for confidence scores
            import matplotlib
            matplotlib.use('TkAgg')
            
            # Close any existing matplotlib figures to prevent multiple windows
            plt.close('all')
            
            # Set dark theme colors
            plt.style.use('dark_background')
            
            # Create new figure
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#2c3e50')
            
            bars = ax.bar(self.category_names, self.last_probabilities, 
                         color='#3498db', edgecolor='white', linewidth=2)
            ax.set_title('Prediction Confidence Scores', fontsize=20, fontweight='bold', 
                        pad=20, color='white')
            ax.set_ylabel('Confidence Score', fontsize=16, color='white')
            ax.set_xlabel('Categories', fontsize=16, color='white')
            ax.tick_params(axis='x', rotation=45, labelsize=12, colors='white')
            ax.tick_params(axis='y', labelsize=12, colors='white')
            
            # Add value labels on bars
            for bar, prob in zip(bars, self.last_probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.3f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=12, color='white')
            
            # Make tick labels white
            for label in ax.get_xticklabels():
                label.set_color('white')
            for label in ax.get_yticklabels():
                label.set_color('white')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create confidence chart: {str(e)}")
    
    def update_metrics_display(self, text):
        """Update metrics display with new text"""
        self.metrics_text.insert(tk.END, text)
        self.metrics_text.see(tk.END)

def create_gui():
    """Create and run the GUI application"""
    # Ensure matplotlib uses proper backend for macOS
    import matplotlib
    matplotlib.use('TkAgg')
    
    try:
        root = tk.Tk()
        NewsClassifierGUI(root)
        root.mainloop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"GUI Error: {e}")
    finally:
        # Ensure clean exit
        import sys
        sys.exit(0)

if __name__ == '__main__':
    # Ask user to choose between GUI and command line
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        create_gui()
    else:
        print("Choose interface:")
        print("1. GUI (recommended)")
        print("2. Command line")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            create_gui()
        else:
            execute_news_classifier()
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        create_gui()
    else:
        print("Choose interface:")
        print("1. GUI (recommended)")
        print("2. Command line")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            create_gui()
        else:
            execute_news_classifier()

