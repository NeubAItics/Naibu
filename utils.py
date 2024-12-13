import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import re

def authenticate_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("model_files/credentials-SC.json", scope)
    client = gspread.authorize(creds)
    return client

def load_users_data():
    client = authenticate_google_sheets()
    sheet = client.open_by_key('1lg4nS6uej52cR4QLAn2ZWMoGARpi7fbU4J0GI3avHqY')  # Replace with your actual spreadsheet ID
    worksheet = sheet.worksheet('users')  # Ensure you have a worksheet named 'users'
    data = worksheet.get_all_records()
    if not data:  # Check if data is empty
        return pd.DataFrame(columns=['Doctor Name', 'Username', 'Password', 'Email'])
    return pd.DataFrame(data)

def save_users_data(df):
    client = authenticate_google_sheets()
    sheet = client.open_by_key('1lg4nS6uej52cR4QLAn2ZWMoGARpi7fbU4J0GI3avHqY')  # Replace with your actual spreadsheet ID
    worksheet = sheet.worksheet('users')  # Ensure you have a worksheet named 'users'
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

def register_user(doctor_name, username, password, email):
    """
    Register a new user and create a new worksheet for the user in Google Sheets with default columns.
    """
    # Check if the password is alphanumeric
    if not re.match(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}$', password):
        return "Password should be alphanumeric and at least 8 characters long!"
    
    try:
        # Load user data from Google Sheets
        users_data = load_users_data()
        
        if username in users_data['Username'].values:
            return "Username already exists!"
        
        # Add new user data
        new_user = pd.DataFrame([[doctor_name, username, password, email]], columns=['Doctor Name', 'Username', 'Password', 'Email'])
        users_data = pd.concat([users_data, new_user], ignore_index=True)
        save_users_data(users_data)
        
        # Create a new worksheet for the user
        client = authenticate_google_sheets()
        sheet = client.open_by_key('1lg4nS6uej52cR4QLAn2ZWMoGARpi7fbU4J0GI3avHqY')  # Replace with your actual spreadsheet ID
        new_worksheet = sheet.add_worksheet(title=username, rows="1000", cols="10")
        
        # Define the default column headers
        column_headers = ['Patient Name', 'Tooth Number', 'Predicted Class', 'Actual Class', 'Opinion', 'Comments']
        
        # Update the first row of the new worksheet with column headers
        new_worksheet.update('A1:F1', [column_headers])
        
        return "Registration successful! Please log in."
    
    except gspread.exceptions.SpreadsheetNotFound:
        return "SpreadsheetNotFound exception caught. The specified spreadsheet was not found."
    except gspread.exceptions.WorksheetNotFound:
        return "WorksheetNotFound exception caught. The specified worksheet was not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def list_spreadsheets():
    """
    List all spreadsheets accessible by the service account.
    """
    client = authenticate_google_sheets()
    spreadsheets = client.list_spreadsheet_files()
    for spreadsheet in spreadsheets:
        print(spreadsheet['name'])

def save_prediction(username, patient_name, tooth_number, predicted_class, actual_class, opinion, comments):
    """
    Save the prediction and feedback to the user's worksheet.
    """
    try:
        # Authenticate and access the Google Sheets
        client = authenticate_google_sheets()
        sheet = client.open_by_key('1lg4nS6uej52cR4QLAn2ZWMoGARpi7fbU4J0GI3avHqY')  # Replace with your actual spreadsheet ID
        worksheet = sheet.worksheet(username)  # Open the user's worksheet
        
        # Prepare the data to be saved
        data = [[patient_name, tooth_number, predicted_class, actual_class, opinion, comments]]
        
        # Append the data to the worksheet
        worksheet.append_rows(data, value_input_option='RAW')
        
    except gspread.exceptions.WorksheetNotFound:
        raise ValueError("Worksheet not found for the user.")
    except Exception as e:
        raise ValueError(f"An error occurred while saving the prediction: {e}")
                                       