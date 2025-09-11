import tensorflow as tf

# モデルをロード
model = tf.keras.models.load_model("c:/stock_app_project/models/stock_predictor_model.h5")

# 構造を表示
model.summary()
