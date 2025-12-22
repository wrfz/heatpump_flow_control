"""Constants for the Wärmepumpen ML integration."""

DOMAIN = "heatpump_flow_control"

# Konfiguration
CONF_AUSSEN_TEMP_SENSOR = "aussen_temp_sensor"
CONF_RAUM_IST_SENSOR = "raum_ist_sensor"
CONF_RAUM_SOLL_SENSOR = "raum_soll_sensor"
CONF_VORLAUF_IST_SENSOR = "vorlauf_ist_sensor"
CONF_VORLAUF_SOLL_ENTITY = "vorlauf_soll_entity"
CONF_MIN_VORLAUF = "min_vorlauf"
CONF_MAX_VORLAUF = "max_vorlauf"
CONF_UPDATE_INTERVAL = "update_interval"
CONF_LEARNING_RATE = "learning_rate"
CONF_TREND_HISTORY_SIZE = "trend_history_size"

# Defaults
DEFAULT_MIN_VORLAUF = 25.0
DEFAULT_MAX_VORLAUF = 55.0
DEFAULT_UPDATE_INTERVAL = 60  # Minuten
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_TREND_HISTORY_SIZE = 12

# Attribute
ATTR_AUSSEN_TEMP = "aussen_temp"
ATTR_RAUM_IST = "raum_ist"
ATTR_RAUM_SOLL = "raum_soll"
ATTR_VORLAUF_IST = "vorlauf_ist"
ATTR_RAUM_ABWEICHUNG = "raum_abweichung"
ATTR_AUSSEN_TREND = "aussen_trend"
ATTR_AUSSEN_TREND_5MIN = "aussen_trend_5min"
ATTR_AUSSEN_TREND_30MIN = "aussen_trend_30min"
ATTR_MODEL_MAE = "model_mae"
ATTR_PREDICTIONS_COUNT = "predictions_count"
ATTR_LAST_UPDATE = "last_update"
ATTR_NEXT_UPDATE = "next_update"

CONF_ACTIVE_SWITCH = "active"
