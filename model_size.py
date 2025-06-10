import os

model_path = "mlartifacts\\603217986484123572\\a92663a6fcdf464f8fd77b20bd3a2ff1\\artifacts\models_mlflow\model.pkl"
model_size_bytes = os.path.getsize(model_path)
print(f"📦 Model size: {model_size_bytes} bytes")
