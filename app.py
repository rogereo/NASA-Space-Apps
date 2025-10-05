from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Columns to hide from the table view
HIDDEN_COLS = {'gif', 'nasa_url', 'educational_summary'}

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

        # Get the page number from the query parameters (default to 1)
        page = int(request.args.get('page', 1))
        records_per_page = 10

        # Calculate the start and end indices for pagination
        start_idx = (page - 1) * records_per_page
        end_idx = start_idx + records_per_page

        # Full columns and visible columns
        all_columns = df.columns.tolist()
        display_columns = [c for c in all_columns if c not in HIDDEN_COLS]

        # Slice the DataFrame for the current page, but keep all original columns
        paginated_data = df.iloc[start_idx:end_idx].to_dict('records')

        # Calculate total pages
        total_pages = (len(df) + records_per_page - 1) // records_per_page

        # Get basic statistics (report visible features)
        stats = {
            'total_rows': len(df),
            'total_columns': len(display_columns),
            'column_names': display_columns,
            'current_page': page,
            'total_pages': total_pages  # Include total_pages in stats
        }

        return render_template('dashboard.html',
                               data=paginated_data,
                               columns=display_columns,
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

@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404

@app.route("/search")
def search():
    try:
        query = request.args.get('q', '').lower()
        df = pd.read_csv('data.csv')
        
        if query:
            # Search across all columns (search uses full data)
            mask = df.astype(str).apply(lambda x: x.str.lower().str.contains(query)).any(axis=1)
            filtered_df = df[mask]
        else:
            filtered_df = df

        # Return the full data for each row, not the filtered version
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
