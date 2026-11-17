"""
Step 0: Reproducible Raw Order-Level Extraction from Hospital HIS-LIS SQL Server.
Extracts 10 months (2026-01-01 to 2026-10-31) of pseudonymized order-level records
from LABLink SQL Server with ZERO simulated or predicted values, including:
  - T2 (T.Test_Date), T3 (L.DeviceReceived_Date), T4 (R.TestDate_Created), T5 (Finalization_Date)
  - Cross-check offsets against T.Require_Date, L.Tube_Collected_Date, and physical barcode
    scan-in at laboratory reception (RECEIVED in HisLis_NamHoc.dbo.DT_SAMPLE_TRACKING_EVENTS).
If running offline without authorized hospital SQL Server access, use the
open synthetic benchmark dataset `data/synthetic_demo_orders.csv.gz` included in
this repository (or provide an authorized institutional extract locally).
"""

import os
import hmac
import hashlib
import json
import time
import pyodbc
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SENDOUT_XLSX = os.path.join(DATA_DIR, "danhmucguimau.xlsx")
OUT_ORDERS_GZ = os.path.join(DATA_DIR, "raw_10months_orders.csv.gz")
OUT_AUDIT_JSON = os.path.join(DATA_DIR, "empirical_10months_audit.json")

# Configure via environment variables on the protected hospital LAN before running:
#   HIS_SQL_SERVER, HIS_SQL_UID, HIS_SQL_PWD, HIS_HMAC_KEY
CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={os.environ.get('HIS_SQL_SERVER', '')};"
    "DATABASE=LABLink;"
    f"UID={os.environ.get('HIS_SQL_UID', '')};"
    f"PWD={os.environ.get('HIS_SQL_PWD', '')};"
    "TrustServerCertificate=yes;"
)


def map_resource_group(test_type_id):
    if pd.isna(test_type_id):
        return "Other"
    tid = int(test_type_id)
    if tid in (1, 10):
        return "Biochemistry"
    if tid == 2:
        return "Hematology"
    if tid == 3:
        return "Urinalysis"
    if tid in (4, 8, 13, 25, 84):
        return "Immunoassay"
    if tid == 6:
        return "Coagulation"
    return "Other"


def extract_all():
    t0 = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    df_so = pd.read_excel(SENDOUT_XLSX)
    sendout_codes = sorted({str(x).strip() for x in df_so["TestData_ID"].dropna().unique() if str(x).strip()})
    print(f"[+] Loaded {len(sendout_codes)} unique TestData_ID codes from {SENDOUT_XLSX}")

    conn = pyodbc.connect(CONN_STR, timeout=120)
    cur = conn.cursor()
    cur.execute("IF OBJECT_ID('tempdb..#SendOutCodes') IS NOT NULL DROP TABLE #SendOutCodes;")
    cur.execute("CREATE TABLE #SendOutCodes (TestData_ID NVARCHAR(100) PRIMARY KEY);")
    cur.executemany("INSERT INTO #SendOutCodes VALUES (?)", [(c,) for c in sendout_codes])
    conn.commit()

    hmac_key = os.environ.get("HIS_HMAC_KEY", "").encode("utf-8")
    if not hmac_key:
        raise RuntimeError("Set HIS_HMAC_KEY environment variable for keyed HMAC-SHA256 pseudonymization.")

    # 1. Extract every order (Test_ID) across 2026-01-01 to 2026-10-31
    sql_orders = """
    WITH RawDetails AS (
        SELECT
            CAST(T.Test_Date AS DATE) AS OrderDate,
            CONVERT(CHAR(7), T.Test_Date, 120) AS YearMonth,
            CONCAT(P.MaLanKham, '|', CONVERT(CHAR(10), T.Test_Date, 120)) AS RawEncounterToken,
            T.Test_ID,
            T.TestType_ID,
            R.Device_ID,
            R.TestDetail_ID,
            CASE WHEN S.TestData_ID IS NOT NULL THEN 1 ELSE 0 END AS IsSendOutRow,
            T.Test_Date AS T2_Time,
            T.Require_Date AS Require_Time,
            L.Tube_Collected_Date AS Collected_Time,
            L.DeviceReceived_Date AS T3_Time,
            COALESCE(TR_BC.Received_Time, TR_KB.Received_Time) AS LabReceived_Time,
            CAST(R.TestDate_Created AS DATETIME) AS T4_Time,
            COALESCE(
                R.Finalization_Date,
                T.Finalization_Date,
                CASE
                    WHEN DATEDIFF(MINUTE, T.Test_Date, P.Finalization_Date) BETWEEN 1 AND 720
                    THEN P.Finalization_Date
                END
            ) AS T5_Time
        FROM LABLink.dbo.L_PATIENT_INFO P WITH (NOLOCK)
        INNER JOIN LABLink.dbo.T_TEST_INFO T WITH (NOLOCK) ON P.Patient_ID = T.Patient_ID
        INNER JOIN LABLink.dbo.T_RESULT_DETAIL R WITH (NOLOCK) ON T.Test_ID = R.Test_ID
        LEFT JOIN (
            SELECT Test_ID, MIN(Tube_Collected_Date) AS Tube_Collected_Date, MIN(DeviceReceived_Date) AS DeviceReceived_Date
            FROM LABLink.dbo.T_REG_LIST WITH (NOLOCK)
            GROUP BY Test_ID
        ) L ON L.Test_ID = T.Test_ID
        LEFT JOIN (
            SELECT LTRIM(RTRIM(Barcode)) AS Barcode, CAST(Event_Time AS DATE) AS EvDate, MIN(Event_Time) AS Received_Time
            FROM HisLis_NamHoc.dbo.DT_SAMPLE_TRACKING_EVENTS WITH (NOLOCK)
            WHERE Check_Point = 'RECEIVED' AND Barcode IS NOT NULL AND Barcode <> ''
            GROUP BY LTRIM(RTRIM(Barcode)), CAST(Event_Time AS DATE)
        ) TR_BC ON TR_BC.Barcode = LTRIM(RTRIM(T.Barcode)) AND TR_BC.EvDate = CAST(T.Test_Date AS DATE)
        LEFT JOIN (
            SELECT LEFT(LTRIM(RTRIM(SO_LAN_KB)), CHARINDEX('_', LTRIM(RTRIM(SO_LAN_KB)) + '_') - 1) AS Base_KB,
                   CAST(Event_Time AS DATE) AS EvDate, MIN(Event_Time) AS Received_Time
            FROM HisLis_NamHoc.dbo.DT_SAMPLE_TRACKING_EVENTS WITH (NOLOCK)
            WHERE Check_Point = 'RECEIVED' AND SO_LAN_KB IS NOT NULL AND SO_LAN_KB <> '' AND SO_LAN_KB <> 'N/A'
            GROUP BY LEFT(LTRIM(RTRIM(SO_LAN_KB)), CHARINDEX('_', LTRIM(RTRIM(SO_LAN_KB)) + '_') - 1), CAST(Event_Time AS DATE)
        ) TR_KB ON TR_KB.Base_KB = LTRIM(RTRIM(P.MaLanKham)) AND TR_KB.EvDate = CAST(T.Test_Date AS DATE)
        LEFT JOIN #SendOutCodes S ON LTRIM(RTRIM(R.TestData_ID)) = S.TestData_ID
        WHERE T.Test_Date >= '2026-01-01 00:00:00'
          AND T.Test_Date <  '2026-11-01 00:00:00'
          AND P.MaLanKham IS NOT NULL
          AND P.MaLanKham <> ''
          AND ISNULL(R.IsCancelled, 0) = 0
    )
    SELECT
        CONVERT(CHAR(10), OrderDate, 120) AS OrderDate,
        YearMonth,
        RawEncounterToken,
        Test_ID,
        CAST(TestType_ID AS INT) AS TestType_ID,
        COALESCE(MAX(Device_ID), -1) AS Device_ID,
        COUNT(*) AS ParamCount,
        MAX(IsSendOutRow) AS IsExternalOrder,
        SUM(IsSendOutRow) AS SendOutDetailRows,
        ROUND(DATEDIFF(SECOND, CAST(OrderDate AS DATETIME), MIN(T2_Time)) / 60.0, 3) AS T2_Min,
        ROUND(DATEDIFF(SECOND, CAST(OrderDate AS DATETIME), MIN(T3_Time)) / 60.0, 3) AS T3_Min,
        ROUND(DATEDIFF(SECOND, CAST(OrderDate AS DATETIME), MAX(T4_Time)) / 60.0, 3) AS T4_Min,
        ROUND(DATEDIFF(SECOND, CAST(OrderDate AS DATETIME), MAX(T5_Time)) / 60.0, 3) AS T5_Raw_Min,
        ROUND(DATEDIFF(SECOND, MIN(Require_Time), MIN(T2_Time)) / 60.0, 3) AS Test_minus_Require_Min,
        ROUND(DATEDIFF(SECOND, MIN(Collected_Time), MIN(T2_Time)) / 60.0, 3) AS Test_minus_Collected_Min,
        ROUND(DATEDIFF(SECOND, MIN(T2_Time), MIN(T3_Time)) / 60.0, 3) AS Device_minus_Test_Min,
        ROUND(DATEDIFF(SECOND, MIN(T2_Time), MIN(LabReceived_Time)) / 60.0, 3) AS Received_minus_Test_Min,
        ROUND(DATEDIFF(SECOND, CAST(OrderDate AS DATETIME), MIN(LabReceived_Time)) / 60.0, 3) AS Received_Min_Of_Day
    FROM RawDetails
    GROUP BY OrderDate, YearMonth, RawEncounterToken, Test_ID, TestType_ID
    ORDER BY OrderDate, T2_Min, Test_ID;
    """
    print("[+] Querying order-level raw records across Jan 1 - Oct 31, 2026...")
    df_orders = pd.read_sql(sql_orders, conn)
    df_orders["EncounterKey"] = [
        hmac.new(hmac_key, tok.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
        for tok in df_orders.pop("RawEncounterToken")
    ]
    df_orders["ResourceGroup"] = df_orders["TestType_ID"].apply(map_resource_group)
    cols = [
        "OrderDate", "YearMonth", "EncounterKey", "Test_ID", "TestType_ID", "Device_ID",
        "ParamCount", "IsExternalOrder", "SendOutDetailRows",
        "T2_Min", "T3_Min", "T4_Min", "T5_Raw_Min", "ResourceGroup",
        "Test_minus_Require_Min", "Test_minus_Collected_Min", "Device_minus_Test_Min",
        "Received_minus_Test_Min", "Received_Min_Of_Day",
    ]
    df_orders = df_orders[cols]

    # 2. Extract raw overall detail audit & empirical Device_ID audit
    sql_audit = """
    SELECT
        COUNT(*) AS raw_detail_rows_all,
        COUNT(DISTINCT R.Test_ID) AS raw_orders_all
    FROM LABLink.dbo.L_PATIENT_INFO P WITH (NOLOCK)
    INNER JOIN LABLink.dbo.T_TEST_INFO T WITH (NOLOCK) ON P.Patient_ID = T.Patient_ID
    INNER JOIN LABLink.dbo.T_RESULT_DETAIL R WITH (NOLOCK) ON T.Test_ID = R.Test_ID
    WHERE T.Test_Date >= '2026-01-01 00:00:00'
      AND T.Test_Date <  '2026-11-01 00:00:00'
      AND P.MaLanKham IS NOT NULL
      AND P.MaLanKham <> ''
    """
    df_tot = pd.read_sql(sql_audit, conn)

    sql_dev = """
    SELECT
        D.Device_ID,
        D.Device_Name,
        D.TestType_ID AS Device_TestType_ID,
        COUNT(DISTINCT CASE WHEN R.Test_Date >= '2026-01-01' AND R.Test_Date < '2026-03-01' THEN R.Test_ID END) AS JanFeb_Orders,
        COUNT(DISTINCT CASE WHEN R.Test_Date >= '2026-05-01' AND R.Test_Date < '2026-06-01' THEN R.Test_ID END) AS May_Orders,
        COUNT(DISTINCT CASE WHEN R.Test_Date >= '2026-06-01' AND R.Test_Date < '2026-07-01' THEN R.Test_ID END) AS Jun_Orders,
        COUNT(DISTINCT CASE WHEN R.Test_Date >= '2026-07-01' AND R.Test_Date < '2026-11-01' THEN R.Test_ID END) AS JulOct_Orders,
        COUNT(DISTINCT CASE WHEN R.Test_Date >= '2026-01-01' AND R.Test_Date < '2026-11-01' THEN R.Test_ID END) AS Total_10M_Orders
    FROM LABLink.dbo.D_DEVICE_LIST D WITH (NOLOCK)
    LEFT JOIN LABLink.dbo.T_RESULT_DETAIL R WITH (NOLOCK)
        ON D.Device_ID = R.Device_ID
        AND R.Test_Date >= '2026-01-01' AND R.Test_Date < '2026-11-01'
        AND ISNULL(R.IsCancelled, 0) = 0
    GROUP BY D.Device_ID, D.Device_Name, D.TestType_ID
    HAVING COUNT(R.TestDetail_ID) > 0
    ORDER BY Total_10M_Orders DESC;
    """
    df_dev = pd.read_sql(sql_dev, conn)
    conn.close()

    # Save compressed CSV & audit JSON
    df_orders.to_csv(OUT_ORDERS_GZ, index=False, compression="gzip", encoding="utf-8")
    inhouse_auto = df_orders[(df_orders["IsExternalOrder"] == 0) & (df_orders["ResourceGroup"] != "Other")]

    audit_info = {
        "extraction_elapsed_sec": round(time.time() - t0, 2),
        "raw_detail_rows_all_status": int(df_tot["raw_detail_rows_all"].iloc[0]),
        "raw_non_cancelled_detail_rows": int(df_orders["ParamCount"].sum()),
        "raw_non_cancelled_orders": int(len(df_orders)),
        "sendout_catalog_unique_codes": len(sendout_codes),
        "sendout_matched_detail_rows": int(df_orders["SendOutDetailRows"].sum()),
        "sendout_matched_orders": int((df_orders["IsExternalOrder"] == 1).sum()),
        "retained_inhouse_auto_jobs": int(len(inhouse_auto)),
        "retained_inhouse_encounter_days": int(inhouse_auto["EncounterKey"].nunique()),
        "active_calendar_days": int(df_orders["OrderDate"].nunique()),
        "empirical_device_audit": df_dev.to_dict(orient="records"),
    }
    with open(OUT_AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_info, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved {len(df_orders):,} raw orders to {OUT_ORDERS_GZ} ({os.path.getsize(OUT_ORDERS_GZ):,} bytes)")
    print(f"       Retained automated in-house jobs: {len(inhouse_auto):,} across {inhouse_auto['EncounterKey'].nunique():,} encounter-days in {df_orders['OrderDate'].nunique()} days.")
    print(f"       Audit JSON saved to {OUT_AUDIT_JSON}")


if __name__ == "__main__":
    extract_all()
