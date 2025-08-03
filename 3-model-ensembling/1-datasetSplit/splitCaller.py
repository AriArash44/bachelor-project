import os
import importlib.util
import logging
from pathlib import Path
from multiprocessing import Process
import pandas as pd

DEVICES_COLS = [
    "fridge.datetime", "fridge.fridge_temperature", "fridge.temp_condition",
    "garage_door.datetime", "garage_door.door_state", "garage_door.sphone_signal",
    "gps_tracker.datetime", "gps_tracker.latitude", "gps_tracker.longitude",
    "modbus.datetime", "modbus.FC1_Read_Input_Register",
    "modbus.FC2_Read_Discrete_Value", "modbus.FC3_Read_Holding_Register",
    "modbus.FC4_Read_Coil", "motion_light.datetime", "motion_light.motion_status",
    "motion_light.light_status", "thermostat.datetime",
    "thermostat.current_temperature", "thermostat.thermostat_status",
    "weather.datetime", "weather.temperature", "weather.pressure",
    "weather.humidity"
]

LINUX_COLS = [
    "disk.RDDSK", "disk.WRDSK", "disk.WCANCL", "disk.DSK",
    "memory.MINFLT", "memory.MAJFLT", "memory.VSTEXT", "memory.VSIZE",
    "memory.RSIZE", "memory.VGROW", "memory.RGROW", "memory.MEM",
    "process.TRUN", "process.TSLPI", "process.TSLPU", "process.POLI",
    "process.NICE", "process.PRI", "process.RTPR", "process.CPUNR",
    "process.Status", "process.EXC", "process.State", "process.CPU",
    "PID", "CMD"
]

NETWORK_COLS = [
    "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service",
    "duration", "src_bytes", "dst_bytes", "conn_state", "missed_bytes",
    "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "dns_query",
    "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD", "dns_RA",
    "dns_rejected", "ssl_version", "ssl_cipher", "ssl_resumed",
    "ssl_established", "ssl_subject", "ssl_issuer", "http_trans_depth",
    "http_method", "http_uri", "http_version", "http_request_body_len",
    "http_response_body_len", "http_status_code", "http_user_agent",
    "http_orig_mime_types", "http_resp_mime_types", "weird_name",
    "weird_addl", "weird_notice"
]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df.columns = df.columns.str.strip()
    return (
        df.loc[:, df.columns.intersection(DEVICES_COLS)],
        df.loc[:, df.columns.intersection(LINUX_COLS)],
        df.loc[:, df.columns.intersection(NETWORK_COLS)],
    )


def _run_driver_by_path(driver_path: Path, csv_path: Path) -> None:
    old_cwd = Path.cwd()
    os.chdir(driver_path.parent)
    try:
        spec = importlib.util.spec_from_file_location("drv_mod", str(driver_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.driver_main([str(csv_path)])
        logging.info(f"Completed: {driver_path.name}")
    finally:
        os.chdir(old_cwd)


def calculate_possibilities(
    dev_df: pd.DataFrame,
    lin_df: pd.DataFrame,
    net_df: pd.DataFrame,
    submodels_dir: Path,
    output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("devices", dev_df), ("linux", lin_df), ("network", net_df)):
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        logging.info(f"Wrote fragment: {path}")
    drivers = {
        "devices": submodels_dir / "devices" / "driver_pkg" / "driver.py",
        "linux": submodels_dir / "linux" / "driver_pkg" / "driver.py",
        "network": submodels_dir / "network" / "driver_pkg" / "driver.py",
    }
    procs = []
    for name, drv in drivers.items():
        p = Process(target=_run_driver_by_path, args=(drv, output_dir / f"{name}.csv"))
        p.start()
        logging.info(f"Started '{name}' (PID {p.pid})")
        procs.append(p)
    for p in procs:
        p.join()
    preds = {}
    for name, drv in drivers.items():
        temp_dir = drv.parent / "temp_files"
        file_map = {"devices": "2-y_pred.csv", "linux": "3-y_pred.csv", "network": "7-y_pred.csv"}
        dfp = pd.read_csv(temp_dir / file_map[name])
        if name == "devices":
            dfp["dos"]  = "NaN"
            dfp["mitm"] = "NaN"
        elif name == "linux":
            dfp["backdoor"]   = "NaN"
            dfp["ransomware"] = "NaN"
        dfp.drop(columns=["predicted"], inplace=True)
        preds[name] = dfp.reindex(sorted(dfp.columns), axis=1)
    for name, dfp in preds.items():
        out = output_dir / f"{name}.csv"
        dfp.to_csv(out, index=False)
        logging.info(f"Saved predictions: {out}")

if __name__ == "__main__": 
    activities = pd.read_csv("../0-testTrainMerge/margedTrain.csv") 
    y = activities["type"] 
    x = activities.drop(columns=["type"]) 
    dev, lin, net = split(x) 
    submodels_dir = Path(__file__).resolve().parent.parent.parent / "2-sub-models"
    output_dir = Path(__file__).resolve().parent
    calculate_possibilities(dev, lin, net, submodels_dir, output_dir)
    y.to_csv("types.csv", index=False)