import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import loadConfig
from src.loader import loadBatches

def run(config):
    train_batches, valid_batches = loadBatches(config["train_file"], is_train=True)
    
    model = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
    model.fit(train_batches[0], train_batches[1])

    pred = model.predict(valid_batches[0])

    acc = accuracy_score(valid_batches[1], pred) * 100
    print(f"Train Accuracy: {acc:.3f}")

    if acc > 95.0:
        joblib.dump(model, "model.pkl")
        print("Model saved as model.pkl")
    else:
        print("Model not saved due to low accuracy")


if __name__ == "__main__":
    config = loadConfig()
    run(config)