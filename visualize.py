import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from data_loader import load_data, reduce_to_2d
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def plot_decision_boundary(model, X, y, title, outpath):
    # assumes X is 2D
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    try:
        Z = model.predict(grid)
    except Exception:
        # if model expects original feature space, this will fail; but we train on scaled data reduced to 2D for visualization
        Z = np.zeros(grid.shape[0])
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.2)
    ax.scatter(X[:,0], X[:,1], c=y, s=20)
    ax.set_title(title)
    fig.savefig(outpath)
    plt.close(fig)

def main(model_dir='outputs', outdir='outputs'):
    # load data and reduce to 2D
    X_train, X_test, y_train, y_test, scaler, feature_names = load_data(scale=True)
    X_train_2d, X_test_2d, pca = reduce_to_2d(X_train, X_test)

    # load models
    linear_path = os.path.join(model_dir, 'svm_linear.joblib')
    rbf_path = os.path.join(model_dir, 'svm_rbf.joblib')
    if not os.path.exists(linear_path) or not os.path.exists(rbf_path):
        raise FileNotFoundError('Model files not found in model_dir. Run train.py first.')

    model_linear = load(linear_path)
    model_rbf = load(rbf_path)

    plot_decision_boundary(model_linear, X_train_2d, y_train, 'SVM Linear (train, 2D PCA)', os.path.join(outdir, 'decision_linear.png'))
    plot_decision_boundary(model_rbf, X_train_2d, y_train, 'SVM RBF (train, 2D PCA)', os.path.join(outdir, 'decision_rbf.png'))

    # Confusion matrix and ROC (using sklearn's display helpers)
    y_pred_lin = model_linear.predict(X_test)
    y_prob_lin = model_linear.predict_proba(X_test)[:,1] if hasattr(model_linear, 'predict_proba') else None
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lin).figure_.savefig(os.path.join(outdir, 'cm_linear.png'))

    if y_prob_lin is not None:
        RocCurveDisplay.from_predictions(y_test, y_prob_lin).figure_.savefig(os.path.join(outdir, 'roc_linear_full.png'))

    y_pred_rbf = model_rbf.predict(X_test)
    y_prob_rbf = model_rbf.predict_proba(X_test)[:,1] if hasattr(model_rbf, 'predict_proba') else None
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rbf).figure_.savefig(os.path.join(outdir, 'cm_rbf.png'))
    if y_prob_rbf is not None:
        RocCurveDisplay.from_predictions(y_test, y_prob_rbf).figure_.savefig(os.path.join(outdir, 'roc_rbf_full.png'))

    print('Visualizations saved to', outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='outputs')
    parser.add_argument('--outdir', type=str, default='outputs')
    args = parser.parse_args()
    main(model_dir=args.model_dir, outdir=args.outdir)
