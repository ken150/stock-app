import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import bcrypt
import stripe
import os
from datetime import datetime, timedelta, date
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import yfinance as yf
from dotenv import load_dotenv

# === プロジェクトルートの絶対パス ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === .env をロード ===
load_dotenv(os.path.join(BASE_DIR, ".env"))

# === DB設定 ===
DB_PATH = os.path.join('/tmp', 'data', 'app.db')
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    is_premium = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

# データベースの初期化
def init_db():
    if not os.path.exists(os.path.dirname(DB_PATH)):
        os.makedirs(os.path.dirname(DB_PATH))
    Base.metadata.create_all(engine)

# === モデルの学習と保存（初回デプロイ時のみ実行） ===
def train_and_save_model():
    print("モデルの学習と保存を開始します...")
    data_dir = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    data_file_path = os.path.join(data_dir, 'N225_stock_data.csv')
    
    print("日経平均株価データを20年間分ダウンロード中...")
    df_download = yf.download('^N225', start='2005-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    if not df_download.empty:
        df_download.to_csv(data_file_path)
        print("データダウンロード完了。")
    else:
        print("データダウンロードに失敗しました。")
        return

    print("モデルを学習しています...")
    try:
        df_train = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
        if 'Close' not in df_train.columns:
            print("データファイルに 'Close' 列がありません。")
            return
            
        df_train['Close'] = pd.to_numeric(df_train['Close'], errors='coerce')
        df_train = df_train.dropna(subset=['Close'])

        if len(df_train) < 60:
            print("学習に十分なデータ（60日分以上）がありません。")
            return

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_train['Close'].values.reshape(-1, 1))

        model_instance = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        model_instance.compile(optimizer='adam', loss='mean_squared_error')
        
        training_data = scaled_data[0:len(scaled_data)-60]
        x_train = []
        y_train = []
        for i in range(60, len(training_data)):
            x_train.append(training_data[i-60:i, 0])
            y_train.append(training_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        model_instance.fit(x_train, y_train, batch_size=1, epochs=1)

        model_dir = os.path.join(BASE_DIR, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_instance.save(os.path.join(model_dir, 'stock_predictor_model.h5'))
        print("モデルが正常に保存されました。")

    except Exception as e:
        print(f"モデルの学習中にエラーが発生しました: {e}")

# === Flask/認証設定 ===
app = Flask(__name__, template_folder='templates')
app.secret_key = os.environ.get("APP_SECRET_KEY","dev-secret-change-me")

# モデルのロード
model_path = os.path.join(BASE_DIR, 'models', 'stock_predictor_model.h5')
model = None
if os.path.exists(model_path):
    print("学習済みモデルをロードしています...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("モデルのロードが完了しました。")
    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}")
else:
    # モデルが存在しない場合のみ学習
    print("モデルファイルが見つからないため、モデルの学習を行います。")
    # train_and_save_model() # ここでは呼び出さない

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

limiter = Limiter(get_remote_address, app=app, default_limits=["200/day"])

# === Stripe設定 ===
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY")
PRICE_ID_MONTHLY = os.environ.get("STRIPE_PRICE_ID_MONTHLY")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")

# === UserMixinラッパー ===
class LoginUser(UserMixin):
    def __init__(self, user: User):
        self.id = str(user.id)
        self.email = user.email
        self.is_premium = user.is_premium

@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    u = db.get(User, int(user_id))
    db.close()
    return LoginUser(u) if u else None

# === 起動時にDB初期化 ===
init_db()

# === 無料ユーザーの利用制限 ===
daily_usage = {}
def can_free_user_predict(user_id):
    key = (user_id, date.today().isoformat())
    if daily_usage.get(key, 0) >= 1:
        return False
    daily_usage[key] = daily_usage.get(key, 0) + 1
    return True

# === 開発者アカウント特別扱い ===
DEV_EMAIL = "kn_0ka23@softbank.ne.jp"
def is_dev_user():
    return current_user.is_authenticated and current_user.email == DEV_EMAIL

# === 認証ルーティング ===
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"].encode()
        db = SessionLocal()
        exists = db.query(User).filter_by(email=email).first()
        if exists:
            db.close()
            return render_template("auth.html", mode="signup", error="既に登録済みのメールです")
        pw_hash = bcrypt.hashpw(password, bcrypt.gensalt()).decode()
        u = User(email=email, password_hash=pw_hash)
        db.add(u)
        db.commit()
        login_user(LoginUser(u))
        db.close()
        return redirect(url_for("index"))
    return render_template("auth.html", mode="signup")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"].encode()
        db = SessionLocal()
        u = db.query(User).filter_by(email=email).first()
        if not u or not bcrypt.checkpw(password, u.password_hash.encode()):
            db.close()
            return render_template("auth.html", mode="login", error="メールまたはパスワードが違います")
        login_user(LoginUser(u))
        db.close()
        return redirect(url_for("index"))
    return render_template("auth.html", mode="login")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# === 課金ルーティング ===
@app.route("/pricing")
@login_required
def pricing():
    return render_template("pricing.html", is_premium=getattr(current_user, "is_premium", False) or is_dev_user())

@app.route("/create-checkout-session", methods=["POST"])
@login_required
def create_checkout_session():
    price_id = os.getenv("STRIPE_PRICE_ID_MONTHLY")
    if not price_id:
        return "Stripe price ID not configured (STRIPE_PRICE_ID_MONTHLY)", 500

    domain = request.host_url.rstrip("/")

    try:
        session_stripe = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{domain}/checkout-success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{domain}/pricing",
            customer_email=current_user.email,
            metadata={"user_id": current_user.id},
            allow_promotion_codes=True,
        )
    except stripe.error.StripeError as e:
        return f"Stripe error: {str(e)}", 500
    except Exception as e:
        return f"Server error: {str(e)}", 500

    return redirect(session_stripe.url, code=303)

@app.route("/checkout-success")
@login_required
def checkout_success():
    return render_template("success.html")

@app.route("/stripe-webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=sig_header, secret=WEBHOOK_SECRET
        )
    except ValueError:
        return "Invalid payload", 400
    except stripe.error.SignatureVerificationError:
        return "Invalid signature", 400

    if event["type"] == "checkout.session.completed":
        session_obj = event["data"]["object"]
        user_id = session_obj.get("metadata", {}).get("user_id")
        if user_id:
            db = SessionLocal()
            u = db.get(User, int(user_id))
            if u:
                u.is_premium = True
                db.commit()
            db.close()

    return "", 200

# 予測対象を日経平均株価に固定
stock_list = [{'コード': 'N225', '銘柄名': '日経平均株価'}]

@app.route('/')
@login_required
def index():
    is_premium = getattr(current_user, "is_premium", False) or is_dev_user()
    return render_template('index.html', stocks=stock_list, stock_data=None, is_premium=is_premium)

@app.route('/predict', methods=['POST'])
@login_required
@limiter.limit("1/day", exempt_when=lambda: getattr(current_user, "is_premium", False) or is_dev_user())
def predict():
    if model is None:
        is_premium = getattr(current_user, "is_premium", False) or is_dev_user()
        return render_template('index.html', prediction="モデルがロードされていません。", stocks=stock_list, is_premium=is_premium)
    
    is_premium = getattr(current_user, "is_premium", False) or is_dev_user()

    if not is_premium and not can_free_user_predict(current_user.id):
        return render_template('index.html',
                               prediction="無料プランの本日の利用回数を超えました。プランをアップグレードしてください。",
                               stocks=stock_list,
                               stock_data=None,
                               is_premium=is_premium)
    
    ticker_symbol = request.form['ticker']
    yfinance_ticker = '^' + ticker_symbol if ticker_symbol == 'N225' else ticker_symbol
    
    period = "90d" if is_premium else "30d"
    data = yf.download(yfinance_ticker, period=period)
    
    if data.empty:
        return render_template('index.html', prediction=f"'{ticker_symbol}'のデータが見つかりませんでした。", stocks=stock_list, is_premium=is_premium)
    
    if isinstance(data['Close'], pd.DataFrame):
        close_prices = data['Close'].iloc[:, 0]
    else:
        close_prices = data['Close']

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))

    if len(scaled_data) < 60:
        return render_template('index.html', prediction="データが少なすぎます。", stocks=stock_list, is_premium=is_premium, stock_data=None)

    last_60_days = scaled_data[-60:]
    X_test = np.array([last_60_days])
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    company_name = "日経平均株価"
    
    last_known_date = data.index[-1].date()
    predicted_date = last_known_date + timedelta(days=1)
    
    if is_premium:
        prediction_text = f"{company_name} の{predicted_date}の予測値: {predicted_stock_price[0][0]:.2f}円"
    else:
        current_price = data['Close'].iloc[-1].item()
        direction = "上昇" if predicted_stock_price[0][0] > current_price else "下落"
        prediction_text = f"{company_name} の{predicted_date}は{direction}傾向です。"

    stock_data = {
        "dates": data.index.strftime("%Y-%m-%d").tolist(),
        "prices": close_prices.round(2).tolist(),
        "predicted": float(predicted_stock_price[0][0])
    }
    
    return render_template('index.html', prediction=prediction_text, stocks=stock_list, stock_data=stock_data, is_premium=is_premium)

if __name__ == '__main__':
    init_db()
    
    # アプリ起動時にモデルファイルが存在するか確認し、なければ学習する
    if not os.path.exists(model_path):
        train_and_save_model()
    
    app.run(debug=True)