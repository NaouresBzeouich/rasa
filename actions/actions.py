from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction
import pandas as pd
import os

class ActionCheckEmployeeInfo(Action):
    def name(self) -> str:
        return "action_check_employee_info"
    
    def load_employee_data(self):
        """Load employee data from CSV file"""
        try:
            # Adjust the path to your CSV file location
            csv_path = "data/naouresBata.csv"  # Update this path
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"Error loading employee data: {e}")
            return None
    
    def find_employee_by_name(self, df, person_name):
        """
        Find employee by first name, last name, or full name
        Returns list of matching employees
        """
        if df is None:
            return []
        
        person_name = person_name.lower().strip()
        matches = []
        
        for _, row in df.iterrows():
            first_name = str(row['FirstName']).lower().strip()
            last_name = str(row['LastName']).lower().strip()
            full_name = f"{first_name} {last_name}"
            
            # Check if person_name matches first name, last name, or full name
            if (person_name == first_name or 
                person_name == last_name or 
                person_name == full_name or
                person_name in full_name):
                matches.append({
                    'employee_id': str(row['Employee ID']),
                    'first_name': row['FirstName'],
                    'last_name': row['LastName'],
                    'full_name': f"{row['FirstName']} {row['LastName']}",
                    'email': row['ADEmail']
                })
        
        return matches
    
    def validate_employee_id(self, df, employee_id, person_name):
        """Validate that the employee_id matches the person_name"""
        if df is None:
            return False, None
        
        try:
            # Find employee by ID
            employee_row = df[df['Employee ID'] == int(employee_id)]
            if employee_row.empty:
                return False, "ID non trouvé dans la base de données"
            
            # Get employee info
            emp_info = employee_row.iloc[0]
            first_name = str(emp_info['FirstName']).lower().strip()
            last_name = str(emp_info['LastName']).lower().strip()
            full_name = f"{first_name} {last_name}"
            person_name_lower = person_name.lower().strip()
            
            # Check if the name matches
            if (person_name_lower == first_name or 
                person_name_lower == last_name or 
                person_name_lower == full_name):
                return True, {
                    'employee_id': str(emp_info['Employee ID']),
                    'first_name': emp_info['FirstName'],
                    'last_name': emp_info['LastName'],
                    'full_name': f"{emp_info['FirstName']} {emp_info['LastName']}",
                    'email': emp_info['ADEmail']
                }
            else:
                actual_name = f"{emp_info['FirstName']} {emp_info['LastName']}"
                return False, f"L'ID {employee_id} correspond à {actual_name}, pas à {person_name}"
        
        except ValueError:
            return False, "L'ID doit être numérique"
        except Exception as e:
            return False, f"Erreur de validation: {str(e)}"
        
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # Get current slots
        person = tracker.get_slot("person")
        employee_id = tracker.get_slot("employee_id")
        previous_person = tracker.get_slot("previous_person")
        
        # Load employee data
        df = self.load_employee_data()
        
        # Determine target action based on last intent
        last_intent = tracker.latest_message.get('intent', {}).get('name')
        intent_to_action = {
            'engagement': 'action_show_engagement_score',
            'predict_absence_next_month': 'action_predict_absence_next_month', 
            'predict_absence_duration': 'action_predict_absence_duration',
            'predict_absence_next_year': 'action_predict_absence_next_year',
            'team_turnover': 'utter_team_turnover',
            'generate_report': 'utter_generate_report'
        }
        
        # Check if person is missing
        if not person:
            dispatcher.utter_message(text="Pour quelle personne voulez-vous des informations?")
            return []
        
        # Check if person changed - ENHANCED LOGIC
        if previous_person and person != previous_person:
            if df is not None:
                # Try to find the new person by name in the dataset
                matches = self.find_employee_by_name(df, person)
                
                if len(matches) == 0:
                    dispatcher.utter_message(text=f"Vous demandez maintenant des informations sur {person}. Aucun employé trouvé avec ce nom. Vérifiez l'orthographe ou fournissez l'ID directement.")
                    return [
                        SlotSet("employee_id", None),
                        SlotSet("previous_person", person)
                    ]
                
                elif len(matches) == 1:
                    # Exactly one match found - auto-fill the information
                    match = matches[0]
                    dispatcher.utter_message(
                        text=f"Vous demandez maintenant des informations sur {match['full_name']} (ID: {match['employee_id']})"
                    )
                    # Auto-set the employee_id and continue
                    return [
                        SlotSet("employee_id", match['employee_id']),
                        SlotSet("previous_person", person),
                        FollowupAction("action_check_employee_info")  # Re-run to complete the flow
                    ]
                
                else:
                    # Multiple matches found - ask for clarification
                    match_list = "\n".join([f"- {m['full_name']} (ID: {m['employee_id']})" for m in matches[:5]])
                    dispatcher.utter_message(
                        text=f"Vous demandez maintenant des informations sur {person}. Plusieurs employés trouvés:\n{match_list}\n\nVeuillez préciser l'ID numérique:"
                    )
                    return [
                        SlotSet("employee_id", None),
                        SlotSet("previous_person", person)
                    ]
            else:
                # Fallback if CSV can't be loaded
                dispatcher.utter_message(text=f"Vous demandez maintenant des informations sur {person}. Quel est l'ID de {person}?")
                return [
                    SlotSet("employee_id", None),
                    SlotSet("previous_person", person)
                ]
        
        # Check if employee_id is missing - NEW ENHANCED LOGIC
        if not employee_id:
            if df is not None:
                # Try to find employee by name in the dataset
                matches = self.find_employee_by_name(df, person)
                
                if len(matches) == 0:
                    dispatcher.utter_message(text=f"Aucun employé trouvé avec le nom '{person}'. Vérifiez l'orthographe ou fournissez l'ID directement.")
                    return [SlotSet("previous_person", person)]
                
                elif len(matches) == 1:
                    # Exactly one match found - auto-fill the information
                    match = matches[0]
                    dispatcher.utter_message(
                        text=f"Employé trouvé: {match['full_name']} (ID: {match['employee_id']})"
                    )
                    # Auto-set the employee_id and continue
                    return [
                        SlotSet("employee_id", match['employee_id']),
                        SlotSet("previous_person", person),
                        FollowupAction("action_check_employee_info")  # Re-run to complete the flow
                    ]
                
                else:
                    # Multiple matches found - ask for clarification
                    match_list = "\n".join([f"- {m['full_name']} (ID: {m['employee_id']})" for m in matches[:5]])
                    dispatcher.utter_message(
                        text=f"Plusieurs employés trouvés avec le nom '{person}':\n{match_list}\n\nVeuillez préciser l'ID numérique:"
                    )
                    return [SlotSet("previous_person", person)]
            else:
                # Fallback if CSV can't be loaded
                dispatcher.utter_message(text=f"Quel est l'ID numérique de {person}?")
                return [SlotSet("previous_person", person)]
        
        # Validate employee_id with the dataset
        if df is not None:
            is_valid, result = self.validate_employee_id(df, employee_id, person)
            
            if not is_valid:
                dispatcher.utter_message(text=f"Erreur: {result}. Veuillez fournir un ID valide:")
                return [SlotSet("employee_id", None)]
            
            # Success - display employee info
            emp_info = result
            dispatcher.utter_message(
                text=f"✅ Employé confirmé: {emp_info['full_name']} (ID: {emp_info['employee_id']})"
            )
        else:
            # Fallback validation if CSV can't be loaded
            if not employee_id.isdigit():
                dispatcher.utter_message(text="L'ID employé doit être numérique. Veuillez réessayer:")
                return [SlotSet("employee_id", None)]
            
            dispatcher.utter_message(text=f"Informations pour {person} (ID: {employee_id}):")
        
        # Continue with the original action
        target_action = intent_to_action.get(last_intent, 'utter_default')
        
        return [
            SlotSet("previous_person", person),
            FollowupAction(target_action)
        ]




from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd
import numpy as np
from tensorflow import keras
from joblib import load


class ActionShowEngagementScore(Action):
    def name(self) -> Text:
        return "action_show_engagement_score"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        employee_id = next(tracker.get_latest_entity_values("employee_id"), None)
        if employee_id is None:
            dispatcher.utter_message(text="Veuillez fournir un identifiant d'employé afin que je puisse vérifier le score d'engagement.")
            return []

        try:
            employee_id = int(employee_id)

            # Load trained LSTM model and scaler
            model = keras.models.load_model("actions/my_model.h5")
            scaler = load("actions/scaler_model.joblib")

            # Load dataset
            df = pd.read_csv("data/naouresBata (1).csv")

            # Preprocessing (same as training)
            df["Survey Date"] = pd.to_datetime(df["Survey Date"])
            df = df.sort_values(by=["Employee ID", "Survey Date"])
            df = df.drop(columns=["Employee ID.1", "Performance"], errors="ignore")

            # Create lag features
            df["Engagement Score Lag1"] = df.groupby("Employee ID")["Engagement Score"].shift(1)
            df["Engagement Score Lag2"] = df.groupby("Employee ID")["Engagement Score"].shift(2)
            df["Engagement Score Lag3"] = df.groupby("Employee ID")["Engagement Score"].shift(3)

            # Fill NaNs with per-employee mean
            employee_means = df.groupby("Employee ID")["Engagement Score"].transform("mean")
            df["Engagement Score Lag1"].fillna(employee_means, inplace=True)
            df["Engagement Score Lag2"].fillna(employee_means, inplace=True)
            df["Engagement Score Lag3"].fillna(employee_means, inplace=True)

            # Filter for the given employee
            emp_df = df[df["Employee ID"] == employee_id]
            if emp_df.empty:
                dispatcher.utter_message(text=f"Aucune donnée trouvée pour l'identifiant d'employé fourni {employee_id}.")
                return []

            latest = emp_df.sort_values(by="Survey Date", ascending=False).iloc[0]

            # ✅ Keep only numeric columns (as used in training)
            numeric_columns = [
                col for col in df.columns
                if col not in ["Employee ID", "Survey Date", "Engagement Score"]
                and pd.api.types.is_numeric_dtype(df[col])
            ]

            # Prepare input
            input_df = pd.DataFrame([latest[numeric_columns]])
            input_scaled = scaler.transform(input_df)
            input_scaled = input_scaled.reshape((1, 1, input_scaled.shape[1]))

            # Predict
            prediction = model.predict(input_scaled)
            score = round(float(prediction[0][0]), 2)

            # ✅ Ensure score is not negative
            if score < 0:
                score = 0.0

            dispatcher.utter_message(
                text=f"Le score d'engagement prédit pour l'employé {employee_id} est de {score}."
            )

        except Exception as e:
            dispatcher.utter_message(text=f"Une erreur est survenue lors de la prédiction du score d'engagement: {str(e)}")

        return []


from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np

# Define focal loss
def focal_loss(gamma=2., alpha=0.25):
    def focal(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.mean(loss, axis=-1)
    return focal
    
class ActionPredictAbsenceNextMonth(Action):
    def name(self) -> Text:
        return "action_predict_absence_next_month"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        employee_id = next(tracker.get_latest_entity_values("employee_id"), None)
        if employee_id is None:
            dispatcher.utter_message(text="Veuillez fournir un identifiant d'employé pour vérifier la prédiction d'absence.")
            return []

        try:
            employee_id = int(employee_id)

            # Load the model with focal loss and scaler
            model = load_model("actions/employee_absenteeism_model.h5",
                               custom_objects={"focal": focal_loss()})
            scaler = load("actions/scaler_abs.joblib")

            # Load dataset
            df = pd.read_csv("data/Employee_Absenteeism_Per_Year.csv")

            # Preprocess
            month_columns = [str(i) for i in range(1, 13)]
            df[month_columns] = df[month_columns].applymap(lambda x: 1 if x > 0 else 0)

            emp_data = df[df["Employee ID"] == employee_id].sort_values(by="Year")

            if emp_data.empty:
                dispatcher.utter_message(text=f"Aucune donnée trouvée pour l'identifiant d'employé fourni {employee_id}.")
                return []

            # Sequence preparation
            monthly_absences = emp_data[month_columns].values.flatten()
            if len(monthly_absences) < 12:
                dispatcher.utter_message(text=f"Pas suffisamment de données pour l'employé {employee_id} afin de faire une prédiction.")
                return []

            latest_sequence = monthly_absences[-12:]
            input_scaled = scaler.transform(latest_sequence.reshape(1, -1)).reshape(1, 12, 1)

            # Predict
            y_pred = model.predict(input_scaled)
            y_pred_bin = (y_pred > 0.5).astype(int)

            # We'll use the first prediction value for "next month"
            next_month_pred = y_pred_bin[0][0]

            if next_month_pred:
                dispatcher.utter_message(text=f"Oui, l'employé {employee_id} est prédit comme absent le mois prochain.")
            else:
                dispatcher.utter_message(text=f"Non, l'employé {employee_id} n'est pas prédit comme absent le mois prochain.")

        except Exception as e:
            dispatcher.utter_message(text=f"Une erreur est survenue lors de la prédiction de l'absence: {str(e)}")

        return []



class ActionPredictAbsence(Action):
    def name(self) -> Text:
        return "action_predict_absence_next_year"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        employee_id = next(tracker.get_latest_entity_values("employee_id"), None)
        if employee_id is None:
            dispatcher.utter_message(text="Veuillez fournir un identifiant d'employé pour vérifier la prédiction d'absence.")
            return []

        try:
            employee_id = int(employee_id)

            # Load the trained model with focal loss and scaler
            model = load_model("actions/employee_absenteeism_model.h5",
                               custom_objects={"focal": focal_loss()})
            scaler = load("actions/scaler_abs.joblib")

            # Load dataset
            df = pd.read_csv("data/Employee_Absenteeism_Per_Year.csv")

            # Preprocess dataset
            month_columns = [str(i) for i in range(1, 13)]
            df[month_columns] = df[month_columns].applymap(lambda x: 1 if x > 0 else 0)

            emp_data = df[df["Employee ID"] == employee_id].sort_values(by="Year")

            if emp_data.empty:
                dispatcher.utter_message(text=f"Aucune donnée trouvée pour l'identifiant d'employé fourni {employee_id}.")
                return []

            # Flatten monthly data to sequence
            monthly_absences = emp_data[month_columns].values.flatten()
            if len(monthly_absences) < 12:
                dispatcher.utter_message(text=f"Pas suffisamment de données pour l'employé {employee_id} afin de faire une prédiction.")
                return []

            latest_sequence = monthly_absences[-12:]
            input_scaled = scaler.transform(latest_sequence.reshape(1, -1)).reshape(1, 12, 1)

            # Predict
            y_pred = model.predict(input_scaled)
            y_pred_bin = (y_pred > 0.5).astype(int)

            will_be_absent = y_pred_bin.sum() > 0

            if will_be_absent:
                dispatcher.utter_message(text=f"Oui, l'employé {employee_id} est prédit comme absent l'année prochaine.")
            else:
                dispatcher.utter_message(text=f"Non, l'employé {employee_id} n'est pas prédit comme absent l'année prochaine.")

        except Exception as e:
            dispatcher.utter_message(text=f"Une erreur est survenue lors de la prédiction de l'absence: {str(e)}")

        return []



from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd
import numpy as np
import joblib
import os

class ActionPredictAbsenceDurationByID(Action):
    def name(self) -> Text:
        return "action_predict_absence_duration"

    def __init__(self):
        model_path = "actions/xgb_model.joblib"
        data_path = "data/AbsenceData2.csv"

        # Load model
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError("Model file not found at actions/xgb_model.joblib")

        # Load dataset
        if os.path.exists(data_path):
            self.data = pd.read_csv(data_path)
        else:
            raise FileNotFoundError("Employee data not found at data/AbsenceData2.csv")

        # List of features used for prediction (exclude target and ID)
        self.features = [
            'Absence Type', 'Day of the Week', 'Week of Year', 'Holiday Proximity',
            'Previous Absence History', 'Weather (Temp)', 'Season',
            'Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score',
            'EmployeeClassificationType', 'DepartmentType', 'MaritalDesc',
            'Current Employee Rating', 'Education', 'JobSatisfaction', 'JobInvolvement',
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
            'TrainingTimesLastYear', 'MonthlyIncome', 'PercentSalaryHike', 'OverTime',
            'DistanceFromHome', 'EducationField', 'Performance', 'YearsWorking',
            'FinalScore', 'ExitScore', 'Age', 'SupervisorEngagementScore',
            'PayZone_Encoded', 'Year', 'Month', 'DayOfWeek', 'DayOfYear',
            'AbsenceType_Season', 'PreviousAbsence_HolidayProximity'
        ]

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        employee_id = tracker.get_slot("employee_id")

        if not employee_id:
            dispatcher.utter_message(text="Veuillez fournir un identifiant d'employé valide, s'il vous plaît.")
            return []

        try:
            employee_id = int(employee_id)
        except ValueError:
            dispatcher.utter_message(text="L'identifiant de l'employé doit être numérique.")
            return []

        emp_data = self.data[self.data["Employee ID"] == employee_id]
        if emp_data.empty:
            dispatcher.utter_message(text=f"Aucune donnée trouvée pour l'identifiant d'employé fourni {employee_id}.")
            return []

        latest = emp_data.sort_values(["Year", "Month"], ascending=False).iloc[0]

        try:
            input_data = latest[self.features].values.reshape(1, -1)
            raw_prediction = self.model.predict(input_data)[0]

            # Round to the nearest whole number (up if >= .5, down otherwise)
            prediction = int(round(np.clip(raw_prediction, 0, 31)))  # cap max at 31 days if needed

            dispatcher.utter_message(
                text = f"📊 La durée d'absence prévue pour l'employé ID {employee_id} est de **{prediction} jours**."
            )

        except Exception as e:
            dispatcher.utter_message(text=f"⚠️ Échec de la prédiction: {str(e)}")

        return []


