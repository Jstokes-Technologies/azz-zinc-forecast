-- Run as SYSDBA
ALTER SESSION SET CONTAINER = FREEPDB1;

-- Tablespace
CREATE TABLESPACE zinc_forecast_ts
  DATAFILE 'zinc_forecast_ts01.dbf'
  SIZE 100M AUTOEXTEND ON NEXT 50M MAXSIZE 2G;

-- User
CREATE USER zinc_forecast
  IDENTIFIED BY "ZincForecast2024!"
  DEFAULT TABLESPACE zinc_forecast_ts
  TEMPORARY TABLESPACE temp
  QUOTA UNLIMITED ON zinc_forecast_ts;

GRANT CREATE SESSION, CREATE TABLE, CREATE VIEW,
      CREATE SEQUENCE, CREATE PROCEDURE, CREATE TRIGGER,
      CREATE TYPE TO zinc_forecast;

ALTER SESSION SET CURRENT_SCHEMA = zinc_forecast;

-- Table: data_sources
CREATE TABLE zinc_forecast.data_sources (
  source_id    NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_name  VARCHAR2(100) NOT NULL,
  source_type  VARCHAR2(50) NOT NULL
               CONSTRAINT chk_source_type CHECK (source_type IN ('FREE_API','LME_PAID','MANUAL')),
  api_endpoint VARCHAR2(500),
  description  VARCHAR2(1000),
  is_active    NUMBER(1) DEFAULT 1 NOT NULL,
  created_date TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

-- Table: zinc_prices
CREATE TABLE zinc_forecast.zinc_prices (
  price_id      NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  price_date    DATE NOT NULL,
  price_usd_mt  NUMBER(12,2) NOT NULL,
  price_usd_lb  NUMBER(10,6),
  source_id     NUMBER NOT NULL REFERENCES zinc_forecast.data_sources(source_id),
  volume        NUMBER(15,2),
  high_usd_mt   NUMBER(12,2),
  low_usd_mt    NUMBER(12,2),
  open_usd_mt   NUMBER(12,2),
  close_usd_mt  NUMBER(12,2),
  load_timestamp TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
  CONSTRAINT uq_zinc_price_date_source UNIQUE (price_date, source_id)
);
CREATE INDEX idx_zinc_prices_date ON zinc_forecast.zinc_prices(price_date);

-- Trigger: auto-calculate price_usd_lb
CREATE OR REPLACE TRIGGER zinc_forecast.trg_calc_price_lb
  BEFORE INSERT OR UPDATE ON zinc_forecast.zinc_prices
  FOR EACH ROW
BEGIN
  :NEW.price_usd_lb := ROUND(:NEW.price_usd_mt / 2204.62, 6);
END;
/

-- Table: forecast_runs
CREATE TABLE zinc_forecast.forecast_runs (
  run_id              NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_timestamp       TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
  model_type          VARCHAR2(50) DEFAULT 'HOLT_WINTERS' NOT NULL,
  seasonality_mode    VARCHAR2(20),
  seasonal_periods    NUMBER,
  horizon_days        NUMBER NOT NULL,
  horizon_label       VARCHAR2(50) NOT NULL,
  training_start_date DATE,
  training_end_date   DATE,
  data_points_used    NUMBER,
  aic_score           NUMBER(15,6),
  mape                NUMBER(8,4),
  rmse                NUMBER(12,4),
  parameters_json     CLOB,
  ai_commentary       CLOB,
  status              VARCHAR2(20) DEFAULT 'RUNNING' NOT NULL
                      CONSTRAINT chk_run_status CHECK (status IN ('RUNNING','COMPLETED','FAILED')),
  error_message       VARCHAR2(4000)
);

-- Table: forecast_results
CREATE TABLE zinc_forecast.forecast_results (
  result_id         NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id            NUMBER NOT NULL REFERENCES zinc_forecast.forecast_runs(run_id) ON DELETE CASCADE,
  forecast_date     DATE NOT NULL,
  predicted_usd_mt  NUMBER(12,2) NOT NULL,
  predicted_usd_lb  NUMBER(10,6),
  lower_80_usd_mt   NUMBER(12,2),
  upper_80_usd_mt   NUMBER(12,2),
  lower_95_usd_mt   NUMBER(12,2),
  upper_95_usd_mt   NUMBER(12,2),
  actual_usd_mt     NUMBER(12,2)
);
CREATE INDEX idx_forecast_results_run ON zinc_forecast.forecast_results(run_id);
CREATE INDEX idx_forecast_results_date ON zinc_forecast.forecast_results(forecast_date);

-- Trigger: auto-calculate predicted_usd_lb
CREATE OR REPLACE TRIGGER zinc_forecast.trg_calc_forecast_lb
  BEFORE INSERT OR UPDATE ON zinc_forecast.forecast_results
  FOR EACH ROW
BEGIN
  :NEW.predicted_usd_lb := ROUND(:NEW.predicted_usd_mt / 2204.62, 6);
END;
/

-- Seed data source
INSERT INTO zinc_forecast.data_sources (source_name, source_type, api_endpoint, description)
VALUES ('Yahoo Finance', 'FREE_API', 'yfinance:ZNC=F', 'CME Zinc Futures via yfinance - tracks LME closely');
COMMIT;
