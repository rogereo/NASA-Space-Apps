from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dashboard")
def dashboard():
    try:
        # Read the CSV file
        df = pd.read_csv('data.csv')
        
        # Convert DataFrame to dictionary for easier handling in template
        data = df.to_dict('records')
        
        # Get column names for the table headers
        columns = df.columns.tolist()
        
        # Get basic statistics
        stats = {
            'total_rows': len(df),
            'total_columns': len(columns),
            'column_names': columns
        }
        
        return render_template('dashboard.html', 
                             data=data, 
                             columns=columns, 
                             stats=stats)
    
    except FileNotFoundError:
        return render_template('dashboard.html', 
                             error="data.csv file not found", 
                             data=[], 
                             columns=[], 
                             stats={})
    except Exception as e:
        return render_template('dashboard.html', 
                             error=f"Error reading CSV file: {str(e)}", 
                             data=[], 
                             columns=[], 
                             stats={})

@app.route("/search")
def search():
    try:
        query = request.args.get('q', '').lower()
        df = pd.read_csv('data.csv')
        
        if query:
            # Search across all columns
            mask = df.astype(str).apply(lambda x: x.str.lower().str.contains(query)).any(axis=1)
            filtered_df = df[mask]
        else:
            filtered_df = df
        
        data = filtered_df.to_dict('records')
        return jsonify({
            'data': data,
            'total_results': len(data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)