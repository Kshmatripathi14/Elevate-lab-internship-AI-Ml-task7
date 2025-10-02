import os, json, argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from joblib import dump
import matplotlib.pyplot as plt

from data_loader import load_data

def make_output_dir(path):
    os.makedirs(path, exist_ok=True)

def run_training(output_dir='outputs', cv_folds=5, random_state=42):
    make_output_dir(output_dir)
    X_train, X_test, y_train, y_test, scaler, feature_names = load_data()

    results = {}

    param_grid_linear = { 'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear'] }
    param_grid_rbf = { 'C': [0.1, 1, 10, 50], 'gamma': ['scale', 0.01, 0.1, 1], 'kernel': ['rbf'] }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    print('Running GridSearchCV for linear kernel...')
    gs_linear = GridSearchCV(SVC(probability=True), param_grid_linear, cv=cv, scoring='accuracy', n_jobs=-1)
    gs_linear.fit(X_train, y_train)
    print('Best linear params:', gs_linear.best_params_)
    best_linear = gs_linear.best_estimator_
    dump(best_linear, os.path.join(output_dir, 'svm_linear.joblib'))

    print('Running GridSearchCV for rbf kernel...')
    gs_rbf = GridSearchCV(SVC(probability=True), param_grid_rbf, cv=cv, scoring='accuracy', n_jobs=-1)
    gs_rbf.fit(X_train, y_train)
    print('Best rbf params:', gs_rbf.best_params_)
    best_rbf = gs_rbf.best_estimator_
    dump(best_rbf, os.path.join(output_dir, 'svm_rbf.joblib'))

    # Evaluate
    for name, model in [('linear', best_linear), ('rbf', best_rbf)]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        except Exception:
            auc = None
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {'accuracy': acc, 'roc_auc': auc, 'report': report, 'confusion_matrix': cm, 'best_params': model.get_params()}

        # Save confusion matrix
        fig, ax = plt.subplots()
        cax = ax.matshow(confusion_matrix(y_test, y_pred))
        fig.colorbar(cax)
        ax.set_title(f'Confusion Matrix ({name})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        for (i, j), val in np.ndenumerate(confusion_matrix(y_test, y_pred)):
            ax.text(j, i, int(val), ha='center', va='center')
        fig.savefig(os.path.join(output_dir, f'confusion_{name}.png'))
        plt.close(fig)

        # ROC curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr)
            ax2.set_title(f'ROC Curve ({name})')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            fig2.savefig(os.path.join(output_dir, f'roc_{name}.png'))
            plt.close(fig2)

    # Cross validation scores for best models on full dataset (train+test for CV)
    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])

    for name, model in [('linear', best_linear), ('rbf', best_rbf)]:
        scores = cross_val_score(model, X_all, y_all, cv=cv, scoring='accuracy', n_jobs=-1)
        results[f'{name}_cv_scores'] = scores.tolist()

    # Save results summary
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('Training complete. Outputs saved to', output_dir)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()
    run_training(output_dir=args.output_dir)
