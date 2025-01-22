from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np
import ta
import time
import config
import logging
from datetime import datetime

def setup_logger():
    # Log dosyası için tarih-saat bazlı isim oluştur
    log_filename = f"bot_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Logger'ı yapılandır
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class BinanceFuturesBot:
    def __init__(self):
        self.logger = setup_logger()
        self.logger.info("Bot başlatılıyor...")
        
        # Binance client'ı başlat
        self.client = Client(config.API_KEY, config.API_SECRET, {"verify": True, "timeout": 20}, tld='com')
        
        # Timestamp sync
        self.sync_time()
        
        # Trading çiftleri ve ayarları
        self.symbols = {
            "XRPUSDT": {
                "quantity_precision": 1,
                "min_qty": 1,
                "min_notional": 5.1,  # Minimum işlem büyüklüğü (USDT)
                "max_leverage": 20,    # Maximum izin verilen kaldıraç
                "default_leverage": 5  # Kullanılacak kaldıraç
            },
            "DASHUSDT": {
                "quantity_precision": 2,
                "min_qty": 0.01,
                "min_notional": 5.1,
                "max_leverage": 20,
                "default_leverage": 5
            },
            "ALGOUSDT": {
                "quantity_precision": 1,
                "min_qty": 1,
                "min_notional": 5.1,
                "max_leverage": 20,
                "default_leverage": 5
            }
        }
        
        self.max_open_trades = 5
        self.open_trades = {}
        
        try:
            for symbol in self.symbols:
                self.logger.info(f"\n{'='*50}\n{symbol} için başlangıç ayarları yapılıyor...")
                
                try:
                    # Önce tüm pozisyonları kapat
                    self.make_request(
                        self.client.futures_cancel_all_open_orders,
                        symbol=symbol
                    )
                    self.logger.info(f"{symbol} için tüm açık emirler iptal edildi")
                    
                    # Marj tipini ISOLATED olarak ayarla
                    try:
                        self.make_request(
                            self.client.futures_change_margin_type,
                            symbol=symbol,
                            marginType='ISOLATED'
                        )
                        self.logger.info(f"{symbol} için marj tipi ISOLATED olarak ayarlandı")
                    except Exception as e:
                        if "No need to change margin type" not in str(e):
                            self.logger.warning(f"{symbol} için marj tipi hatası: {e}")
                    
                    # Kaldıracı ayarla
                    leverage = self.symbols[symbol]["default_leverage"]
                    self.make_request(
                        self.client.futures_change_leverage,
                        symbol=symbol,
                        leverage=leverage
                    )
                    self.logger.info(f"{symbol} için kaldıraç {leverage}x olarak ayarlandı")
                    
                    # Pozisyon modunu kontrol et
                    try:
                        position_mode = self.make_request(
                            self.client.futures_get_position_mode
                        )
                        self.logger.info(f"Mevcut pozisyon modu: {position_mode}")
                    except Exception as e:
                        self.logger.warning(f"Pozisyon modu kontrolü hatası: {e}")
                    
                except Exception as e:
                    self.logger.error(f"{symbol} için ayarlama hatası: {e}")
                    continue
                
        except Exception as e:
            self.logger.error(f"Başlangıç ayarları genel hatası: {e}")
    
    def sync_time(self):
        try:
            # Sunucu zamanını al
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)
            self.time_offset = server_time['serverTime'] - local_time
            
            # Client'a timestamp offset'i ayarla
            self.client.timestamp_offset = self.time_offset
            
            self.logger.info(f"Zaman senkronizasyonu yapıldı. Offset: {self.time_offset}ms")
            return True
        except Exception as e:
            self.logger.error(f"Zaman senkronizasyonu hatası: {e}")
            return False

    def make_request(self, func, *args, **kwargs):
        """Güvenli request wrapper"""
        max_retries = 3
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "Timestamp for this request" in str(e):
                    self.logger.warning(f"Timestamp hatası, yeniden senkronize ediliyor... (Deneme {i+1}/{max_retries})")
                    self.sync_time()
                    continue
                raise e
        raise Exception(f"Maximum retry attempts ({max_retries}) exceeded")

    def get_historical_data(self, symbol):
        # Güvenli request kullanımı
        klines = self.make_request(
            self.client.futures_klines,
            symbol=symbol,
            interval='1m',
            limit=200
        )
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        
        # Tüm numerik kolonları dönüştür
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        return df
    
    def calculate_signals(self, df, symbol):
        self.logger.info(f"\n{'-'*50}\n{symbol} için sinyal hesaplanıyor...")
        signals = {}
        
        try:
            # 1. Range Trading Stratejisi
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            range_high = df['high'].rolling(20).max()
            range_low = df['low'].rolling(20).min()
            
            # NaN değerleri temizle
            df = df.fillna(0)
            
            in_range = (df['close'] > range_low) & (df['close'] < range_high)
            signals['range_long'] = (df['close'] - range_low) <= (df['atr'] * 0.5)
            signals['range_short'] = (range_high - df['close']) <= (df['atr'] * 0.5)
            
            # 2. Breakout Trading Stratejisi
            df['sma20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            volume_ma = df['volume'].rolling(20).mean()
            
            breakout_up = (df['close'] > range_high) & (df['close'] > df['sma20'])
            breakout_down = (df['close'] < range_low) & (df['close'] < df['sma20'])
            
            signals['breakout_long'] = breakout_up & (df['volume'] > volume_ma * 1.5)
            signals['breakout_short'] = breakout_down & (df['volume'] > volume_ma * 1.5)
            
            # 3. RSI + Bollinger Bands Stratejisi
            rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            bollinger = ta.volatility.BollingerBands(df['close'], window=20)
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            signals['rsi_bb_long'] = (rsi < 30) & (df['close'] < df['bb_lower'])
            signals['rsi_bb_short'] = (rsi > 70) & (df['close'] > df['bb_upper'])
            
            # NaN değerleri False olarak değiştir
            for key in signals:
                signals[key] = signals[key].fillna(False)
            
            # Son değerleri al
            final_signals = {k: bool(v.iloc[-1]) for k, v in signals.items()}
            
            # Sinyalleri logla
            for strategy, signal in final_signals.items():
                if signal:
                    self.logger.info(f"{symbol} - {strategy} sinyali tespit edildi!")
            
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Sinyal hesaplama hatası ({symbol}): {e}")
            return {k: False for k in ['range_long', 'range_short', 'breakout_long', 'breakout_short', 'rsi_bb_long', 'rsi_bb_short']}
    
    def execute_trade(self, symbol, side, strategy):
        if len(self.open_trades) >= self.max_open_trades:
            self.logger.warning(f"Maksimum açık işlem sayısına ulaşıldı ({self.max_open_trades})")
            return
            
        if symbol in self.open_trades:
            self.logger.warning(f"{symbol} için zaten açık pozisyon var")
            return
            
        try:
            self.logger.info(f"\n{'-'*50}\n{symbol} için {strategy} stratejisi ile {side} işlemi açılıyor...")
            
            # İşlem öncesi ISOLATED margin type kontrolü
            try:
                position_info = self.make_request(
                    self.client.futures_get_position_mode
                )
                
                # Margin type'ı ISOLATED olarak ayarla
                try:
                    self.make_request(
                        self.client.futures_change_margin_type,
                        symbol=symbol,
                        marginType='ISOLATED'
                    )
                    self.logger.info(f"{symbol} için margin type ISOLATED olarak ayarlandı")
                except Exception as e:
                    if "No need to change margin type" not in str(e):
                        self.logger.error(f"Margin type değiştirme hatası: {e}")
                        return
                
                # Kaldıracı kontrol et ve ayarla
                leverage = self.symbols[symbol]["default_leverage"]
                self.make_request(
                    self.client.futures_change_leverage,
                    symbol=symbol,
                    leverage=leverage
                )
                self.logger.info(f"{symbol} için kaldıraç {leverage}x olarak ayarlandı")
                
            except Exception as e:
                self.logger.error(f"Pozisyon modu kontrolü hatası: {e}")
                return
            
            # Güncel fiyatı al
            current_price = float(self.make_request(
                self.client.futures_symbol_ticker, symbol=symbol
            )['price'])
            
            # Minimum işlem büyüklüğünü kontrol et
            min_notional = self.symbols[symbol]["min_notional"]
            min_qty = self.symbols[symbol]["min_qty"]
            
            # Quantity hesapla (minimum gereksinimleri karşılayacak şekilde)
            quantity = max(
                min_qty,
                round(min_notional / current_price, self.symbols[symbol]["quantity_precision"])
            )
            
            # İşlem büyüklüğünü kontrol et
            notional_value = quantity * current_price
            
            self.logger.info(f"Hesaplanan işlem miktarı: {quantity} {symbol} "
                            f"(Gerçek: {notional_value:.2f} USDT)")
            
            if notional_value < min_notional:
                self.logger.error(f"{symbol} için minimum işlem büyüklüğü sağlanamıyor. "
                                f"Minimum: {min_notional} USDT, Mevcut: {notional_value:.2f} USDT")
                return
            
            # Stop loss ve take profit ayarları (strateji bazlı)
            if strategy in ['range_long', 'range_short']:
                sl_percentage = 0.01
                tp_percentage = 0.02
            else:
                sl_percentage = 0.02
                tp_percentage = 0.03
            
            # Stop loss ve take profit fiyatlarını hesapla
            if side == "BUY":
                stop_price = round(current_price * (1 - sl_percentage), 4)
                take_profit = round(current_price * (1 + tp_percentage), 4)
            else:
                stop_price = round(current_price * (1 + sl_percentage), 4)
                take_profit = round(current_price * (1 - tp_percentage), 4)
            
            self.logger.info(f"Giriş Fiyatı: {current_price}")
            self.logger.info(f"Stop Loss: {stop_price}")
            self.logger.info(f"Take Profit: {take_profit}")
            
            # Ana pozisyonu aç
            order = self.make_request(
                self.client.futures_create_order,
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            # Stop loss emri
            self.make_request(
                self.client.futures_create_order,
                symbol=symbol,
                side="SELL" if side == "BUY" else "BUY",
                type='STOP_MARKET',
                stopPrice=stop_price,
                quantity=quantity,
                reduceOnly=True
            )
            
            # Take profit emri
            self.make_request(
                self.client.futures_create_order,
                symbol=symbol,
                side="SELL" if side == "BUY" else "BUY",
                type='TAKE_PROFIT_MARKET',
                stopPrice=take_profit,
                quantity=quantity,
                reduceOnly=True
            )
            
            # Açık işlemi kaydet
            self.open_trades[symbol] = {
                'side': side,
                'strategy': strategy,
                'entry_price': current_price,
                'quantity': quantity,
                'time': time.time()
            }
            
            self.logger.info(f"İşlem başarıyla açıldı:\n"
                           f"Symbol: {symbol}\n"
                           f"Strateji: {strategy}\n"
                           f"Yön: {side}\n"
                           f"Miktar: {quantity}\n"
                           f"Giriş Fiyatı: {current_price}\n"
                           f"Stop Loss: {stop_price}\n"
                           f"Take Profit: {take_profit}")
            
        except Exception as e:
            self.logger.error(f"{symbol} için işlem hatası: {e}")
    
    def check_open_positions(self):
        try:
            positions = self.make_request(self.client.futures_position_information)
            active_positions = {}
            
            self.logger.info("\nAçık Pozisyonlar Kontrol Ediliyor...")
            
            for position in positions:
                amt = float(position['positionAmt'])
                if amt != 0:
                    symbol = position['symbol']
                    active_positions[symbol] = {
                        'amount': amt,
                        'entry_price': float(position['entryPrice']),
                        'unrealized_pnl': float(position['unRealizedProfit'])
                    }
                    self.logger.info(f"Aktif Pozisyon: {symbol}, "
                                   f"Miktar: {amt}, "
                                   f"Giriş: {position['entryPrice']}, "
                                   f"PNL: {position['unRealizedProfit']}")
            
            # open_trades'i güncelle
            self.open_trades = active_positions
            
        except Exception as e:
            self.logger.error(f"Pozisyon kontrolü hatası: {e}")
    
    def run(self):
        self.logger.info("\nBot çalışmaya başladı...")
        while True:
            try:
                self.check_open_positions()
                
                self.logger.info(f"\nMevcut açık işlem sayısı: {len(self.open_trades)}")
                
                for symbol in self.symbols:
                    if len(self.open_trades) >= self.max_open_trades:
                        self.logger.info("Maksimum açık işlem sayısına ulaşıldı, yeni işlem açılmayacak")
                        break
                        
                    df = self.get_historical_data(symbol)
                    signals = self.calculate_signals(df, symbol)
                    
                    current_price = float(self.client.futures_symbol_ticker(symbol=symbol)['price'])
                    self.logger.info(f"{symbol} Güncel Fiyat: {current_price}")
                    
                    # Sinyalleri kontrol et ve işlem aç
                    for strategy, signal in signals.items():
                        if signal:
                            side = "BUY" if "long" in strategy else "SELL"
                            self.execute_trade(symbol, side, strategy)
                            break
                    
                    time.sleep(0.5)  # Her coin arası 0.5 saniye bekle
                
                self.logger.info("2 saniye bekleniyor...\n" + "="*50)
                time.sleep(2)  # Ana döngü için 2 saniye bekle
                
            except Exception as e:
                self.logger.error(f"Genel hata: {e}")
                time.sleep(2)  # Hata durumunda da 2 saniye bekle 
