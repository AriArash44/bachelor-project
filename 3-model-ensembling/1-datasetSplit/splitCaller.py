import os
import importlib.util
import pandas as pd
from pathlib import Path
from multiprocessing import Process

def _run_driver_by_path(driver_path: str, csv_path: str):
    driver_py = Path(driver_path)
    pkg_dir = driver_py.parent
    old_cwd = Path.cwd()
    os.chdir(pkg_dir)
    try:
        spec = importlib.util.spec_from_file_location("drv_mod", str(driver_py))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.driver_main([csv_path])
    finally:
        os.chdir(old_cwd)

def split(df: pd.DataFrame):
    df.columns = df.columns.str.strip()
    return (
        df.loc[:, df.columns.intersection(devices_cols)],
        df.loc[:, df.columns.intersection(linux_cols)],
        df.loc[:, df.columns.intersection(network_cols)],
    )

def calculatePossibilities(dev_df, lin_df, net_df):
    root = Path.cwd()
    dev_csv = root / "devices.csv"
    lin_csv = root / "linux.csv"
    net_csv = root / "network.csv"
    dev_df.to_csv(dev_csv, index=False)
    lin_df.to_csv(lin_csv, index=False)
    net_df.to_csv(net_csv, index=False)
    procs = []
    for driver_py, csv_path in [
        (devices_py, dev_csv),
        (linux_py, lin_csv),
        (network_py, net_csv)
    ]:
        p = Process(
            target=_run_driver_by_path,
            args=(str(driver_py), str(csv_path))
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    dev_pred = pd.read_csv(devices_dir / "temp_files" / "2-y_pred.csv")
    lin_pred = pd.read_csv(linux_dir / "temp_files" / "3-y_pred.csv")
    net_pred = pd.read_csv(network_dir / "temp_files" / "7-y_pred.csv")
    dev_pred["dos"]   = "NaN"
    dev_pred["mitm"]  = "NaN"
    lin_pred["backdoor"]   = "NaN"
    lin_pred["ransomware"] = "NaN"
    for df in (dev_pred, lin_pred, net_pred):
        df.drop(columns=["predicted"], inplace=True)
    dev_pred = dev_pred[sorted(dev_pred.columns)]
    lin_pred = lin_pred[sorted(lin_pred.columns)]
    net_pred = net_pred[sorted(net_pred.columns)]
    dev_pred.to_csv("devices.csv", index=False)
    lin_pred.to_csv("linux.csv", index=False)
    net_pred.to_csv("network.csv", index=False)

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[3] / "2-sub-models"
    devices_py = BASE / "devices" / "driver_pkg" / "driver.py"
    linux_py = BASE / "linux" / "driver_pkg" / "driver.py"
    network_py = BASE / "network" / "driver_pkg" / "driver.py"
    devices_dir = devices_py.parent
    linux_dir = linux_py.parent
    network_dir = network_py.parent
    devices_cols = [ 
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
    linux_cols = [
        "disk.RDDSK", "disk.WRDSK", "disk.WCANCL", "disk.DSK",
        "memory.MINFLT", "memory.MAJFLT", "memory.VSTEXT", "memory.VSIZE",
        "memory.RSIZE", "memory.VGROW", "memory.RGROW", "memory.MEM",
        "process.TRUN", "process.TSLPI", "process.TSLPU", "process.POLI",
        "process.NICE", "process.PRI", "process.RTPR", "process.CPUNR",
        "process.Status", "process.EXC", "process.State", "process.CPU",
        "PID", "CMD"
    ]
    network_cols = [
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
    activities = pd.read_csv("../../0-testTrainMerge/margedTrain.csv")
    y = activities["type"]
    x = activities.drop(columns=["type"])
    dev, lin, net = split(x)
    calculatePossibilities(dev, lin, net)
    y.to_csv("types.csv", index=False)
