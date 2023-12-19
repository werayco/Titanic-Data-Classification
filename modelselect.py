from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



def best_model(x_train, y_train, x_test, y_test, models, params):
    model_plus_scores = {}

    for model_name, model in models.items():
            
            # Extract parameters specific to the current model
            model_params = params.get(model_name) 

            gs = GridSearchCV(model, model_params, cv=3)
            gs.fit(x_train, y_train)

            # Set best parameters found during grid search
            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            # Evaluate the model on test data
            y_pred = model.predict(x_test)
            
            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

            model_plus_scores[model_name] = accuracy

            return model_plus_scores