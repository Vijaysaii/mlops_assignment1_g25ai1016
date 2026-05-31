from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess_data, train_model, evaluate_model


def main():
    print("Loading Boston Housing dataset...")
    df = load_data()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training KernelRidge model...")
    model = KernelRidge(alpha=1.0, kernel='rbf')
    trained_model = train_model(model, X_train, y_train)

    mse = evaluate_model(trained_model, X_test, y_test)
    print(f"Average MSE score on test set (KernelRidge): {mse:.4f}")


if __name__ == "__main__":
    main()
