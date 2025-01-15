import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
from st_aggrid import AgGrid
import matplotlib.pyplot as plt
from datetime import datetime


# ==========================
# 페이지 설정
# ==========================
st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

# ==========================
# 대시보드 제목 추가
# ==========================
st.markdown(
    """
    <div style="padding: 10px; text-align: left;">
        <h1 style="color: #8CC63F; font-family: 'Arial', sans-serif; margin-bottom: 0;">Vision Flow_Demand Forecast</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================
# 사이드바에 탭 구성
# ==========================
with st.sidebar:
    tab_selection = st.radio("Navigator", ["Data preparation", "Sales prediction"])

# ==========================
# 데이터 초기화
# ==========================
if "sales_data" not in st.session_state:
    st.session_state.sales_data = None
if "climate_data" not in st.session_state:
    st.session_state.climate_data = None
if "econ_data" not in st.session_state:
    st.session_state.econ_data = None

# ==========================
# 열 순서 동기화 함수 정의
# ==========================
def align_columns(train_data, predict_data):
    """Align the columns of train and prediction datasets, keeping 'sales' intact."""
    # 모든 열 이름 소문자로 변환
    train_data.columns = train_data.columns.str.lower()
    predict_data.columns = predict_data.columns.str.lower()

    # 'sales' 열 보존
    sales_column = None
    if 'sales' in train_data.columns:
        sales_column = train_data['sales']
        train_data = train_data.drop(columns=['sales'])

    # 공통 열 가져오기
    common_columns = list(set(train_data.columns).intersection(set(predict_data.columns)))

    # 공통 열로 데이터프레임 정렬
    train_data = train_data[sorted(common_columns)]
    predict_data = predict_data[sorted(common_columns)]

    # 'sales' 열 복구
    if sales_column is not None:
        train_data['sales'] = sales_column

    return train_data, predict_data


# ==========================
# 탭 1: Data Preparation
# ==========================

if tab_selection == "Data preparation": 
    data_tabs = st.tabs([
        "Data cleaning & transformation", 
        "Data integration", 
        "Preparing a prediction template", 
        "Variable details"
    ])


    # ==========================
    # 1st 서브탭: Data Cleaning & Transformation
    # ==========================
    with data_tabs[0]:
        st.header("Data cleaning & transformation")    
        sales_col, climate_col, econ_col = st.columns(3)

        # ==========================
        # 1.1. Sales Data
        # ==========================
        with sales_col:
            st.subheader("Sales data")  

            sales_file = st.file_uploader("Upload Sales CSV", type=["csv"], key="sales_file")

            if sales_file:
                try:
                    # CSV 파일 읽기
                    sales_data = pd.read_csv(sales_file, encoding="cp949")

                    # 소문자 변환
                    sales_data.columns = sales_data.columns.str.lower()

                    # 필수 열 검증
                    required_columns = ["date", "sku", "sales"]
                    missing_columns = [col for col in required_columns if col not in sales_data.columns]
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                        st.stop()

                    # 'date' 열 처리
                    sales_data["date"] = pd.to_datetime(sales_data["date"], errors="coerce")
                    sales_data = sales_data.sort_values(by="date").dropna(subset=["date"])
                    st.session_state.sales_data = sales_data

                    # SKU 열 추출 및 저장
                    st.session_state["sku_list"] = sales_data["sku"].unique().tolist()
                    st.success(f"SKU column detected! {len(st.session_state['sku_list'])} unique SKUs saved.")

                    # Raw Sales Data 미리보기
                    st.write("Raw Sales Data:")
                    st.dataframe(sales_data.head(10))

                    # Clean Data 버튼
                    if st.button("Clean sales Data"):
                        try:
                            # 원본 데이터 복사
                            sales_data = st.session_state.sales_data.copy()

                            # 'date'와 'sku' 조합으로 그룹화하여 중복 처리
                            grouped_data = sales_data.groupby(["date", "sku"], as_index=False).sum()

                            # 모든 날짜 범위 생성
                            date_range = pd.date_range(
                                start=grouped_data["date"].min(),
                                end=grouped_data["date"].max(),
                                freq="D"
                            )

                            # 날짜와 SKU의 모든 조합 생성
                            unique_skus = sorted(grouped_data["sku"].unique())  # SKU 정렬
                            complete_index = pd.MultiIndex.from_product([date_range, unique_skus], names=["date", "sku"])

                            # DataFrame을 MultiIndex에 맞게 재구성
                            complete_sales_data = (
                                grouped_data
                                .set_index(["date", "sku"])
                                .reindex(complete_index, fill_value=0)
                                .reset_index()
                            )

                            # 컬럼 이름 복원
                            complete_sales_data.rename(columns={0: "sales"}, inplace=True)

                            # 정리된 데이터 저장
                            st.session_state.sales_data = complete_sales_data
                            st.success("Sales Data cleaned successfully!")

                            # Cleaned Data 미리보기
                            st.write("Cleaned Sales Data:")
                            st.dataframe(st.session_state.sales_data.head(10))
                        except Exception as e:
                            st.error(f"Error cleaning Sales Data: {e}")

                    # 다운로드 버튼
                    if st.session_state.sales_data is not None:
                        csv = st.session_state.sales_data.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="Download Cleaned Sales Data",
                            data=csv,
                            file_name="cleaned_sales_data.csv",
                            mime="text/csv",
                            key="cleaned_sales_data_download"
                        )                   

                    # Integration 탭으로 데이터 전송 버튼
                    if st.button("Send to Data Integration", key="send_sales_to_integration"):
                        st.session_state.sales_data_integration = st.session_state.sales_data
                        st.success("Sales Data sent to Data Integration!")

                except Exception as e:
                    st.error(f"An error occurred: {e}")



        # ==========================
        # 1.2. Climate Data
        # ==========================

        with climate_col:
            st.subheader("Climate data")

            climate_file = st.file_uploader("Upload Climate CSV", type="csv", key="climate_file")

            if climate_file:
                try:
                    # 파일 읽기
                    climate_data = pd.read_csv(climate_file, encoding="utf-8")
                    climate_data.columns = climate_data.columns.str.lower()  # 컬럼 소문자 변환
                    st.session_state.climate_data = climate_data  # 세션 상태에 저장
                except UnicodeDecodeError:
                    try:
                        # 두 번째 시도: CP949 인코딩
                        climate_data = pd.read_csv(climate_file, encoding="cp949")
                        climate_data.columns = climate_data.columns.str.lower()  # 컬럼 소문자 변환
                        st.session_state.climate_data = climate_data  # 세션 상태에 저장
                    except UnicodeDecodeError as e:
                        st.error(f"File encoding error: {e}. Please ensure the file is UTF-8 or CP949 encoded.")
                        st.stop()

                # 데이터 미리보기
                st.success("Climate Data Uploaded Successfully!")
                st.write("Raw Data Preview (First 10 Rows):")
                st.dataframe(st.session_state.climate_data.head(10))

                # 필수 열 검증
                required_columns = ['temp_avg', 'temp_max', 'temp_min', 'wind_avg', 'cloud_avg', 'hum_avg', 'precip_mm', 'snow_depth_max', 'pres_avg']

                # 컬럼 이름을 소문자로 변환
                st.session_state.climate_data.columns = st.session_state.climate_data.columns.str.lower()

                # 소문자로 변환된 상태에서 필수 열 검증
                missing_columns = [col for col in required_columns if col not in st.session_state.climate_data.columns]

                if missing_columns:
                    st.error(f"Missing columns in climate data: {', '.join(missing_columns)}")
                else:
                    # 변수 생성 및 데이터 전송 함수
                    def generate_and_send_climate_data():
                        try:
                            # 누락된 날짜 채우기
                            climate_data = st.session_state.climate_data.copy()
                            climate_data['date'] = pd.to_datetime(climate_data['date'], errors='coerce')
                            climate_data = climate_data.dropna(subset=['date'])  # 유효하지 않은 날짜 제거

                            # 모든 날짜를 포함하도록 보정
                            full_date_range = pd.date_range(start=climate_data['date'].min(), end=climate_data['date'].max(), freq='D')
                            climate_data = climate_data.set_index('date').reindex(full_date_range).reset_index()
                            climate_data.rename(columns={'index': 'date'}, inplace=True)

                            # Step 2: 결측값 처리
                            st.write("Handling missing values...")

                            # Step 2: 결측값 처리
                            climate_data['temp_avg'].fillna(climate_data['temp_avg'].mean(), inplace=True)  # 평균값으로 대체
                            climate_data['temp_min'].fillna(climate_data['temp_min'].mean(), inplace=True)
                            climate_data['temp_max'].fillna(climate_data['temp_max'].mean(), inplace=True)
                            climate_data['precip_mm'].fillna(0, inplace=True)  # 강수량은 0으로 대체
                            climate_data['snow_depth_max'].fillna(0, inplace=True)  # 눈 깊이도 0으로 대체
                            climate_data['wind_avg'].fillna(climate_data['wind_avg'].mean(), inplace=True)
                            climate_data['hum_avg'].fillna(climate_data['hum_avg'].mean(), inplace=True)
                            climate_data['pres_avg'].fillna(climate_data['pres_avg'].mean(), inplace=True)
                            climate_data['cloud_avg'].fillna(climate_data['cloud_avg'].mean(), inplace=True)

                            # 결측값 상태 확인
                            st.write("Missing values after filling:")
                            st.dataframe(climate_data.isnull().sum())  # 결측값 개수 확인

                            # 파생 변수 생성
                            climate_data = st.session_state.climate_data.copy()

                            # 1. 기본 Derived Variables 생성
                            climate_data["temp_range"] = climate_data["temp_max"] - climate_data["temp_min"]
                            climate_data["temp_perceived"] = (
                                13.12 + 0.6215 * climate_data["temp_avg"]
                                - 11.37 * climate_data["wind_avg"] ** 0.16
                                + 0.3965 * climate_data["temp_avg"] * climate_data["wind_avg"] ** 0.16
                            )
                            climate_data["pressure_humidity_index"] = climate_data["pres_avg"] / climate_data["hum_avg"]
                            climate_data["wind_cloud_interaction"] = climate_data["wind_avg"] * climate_data["cloud_avg"]
                            climate_data["rain_indicator"] = (climate_data["precip_mm"] > 0).astype(int)
                            climate_data["rain_wind_interaction"] = climate_data["precip_mm"] * climate_data["wind_avg"]
                            climate_data["rain_cloud_interaction"] = climate_data["precip_mm"] * climate_data["cloud_avg"]
                            climate_data["discomfort_index"] = (
                                0.81 * climate_data["temp_avg"]
                                + 0.01 * climate_data["hum_avg"] * (0.99 * climate_data["temp_avg"] - 14.3) + 46.3
                            )
                            climate_data["snow_indicator"] = (climate_data["snow_depth_max"] > 0).astype(int)

                            # 2. 추가 Derived Variables 생성
                            climate_data["discomfort_temp_interaction"] = climate_data["discomfort_index"] * climate_data["temp_avg"]
                            climate_data["apparent_humidity_interaction"] = climate_data["temp_perceived"] * climate_data["hum_avg"]

                            # 3. Secondary Derived Variables 생성
                            climate_data["temp_precip_interaction"] = climate_data["temp_avg"] * climate_data["precip_mm"]
                            climate_data["humidity_wind_interaction"] = climate_data["hum_avg"] * climate_data["wind_avg"]
                            climate_data["temp_cloud_interaction"] = climate_data["temp_avg"] * climate_data["cloud_avg"]
                            climate_data["wind_temp_range_interaction"] = climate_data["wind_avg"] * climate_data["temp_range"]
                            climate_data["precip_discomfort_interaction"] = climate_data["precip_mm"] * climate_data["discomfort_index"]
                            climate_data["pressure_temp_interaction"] = climate_data["pres_avg"] * climate_data["temp_avg"]
                            climate_data["humidity_temp_range_interaction"] = climate_data["hum_avg"] * climate_data["temp_range"]

                            # 최신 데이터 업데이트
                            st.session_state.climate_data = climate_data.copy()

                            # 데이터 Integration으로 전송
                            st.session_state.climate_data_integration = st.session_state.climate_data.copy()
                            st.success("Climate Variables Generated and sent to Data Integration successfully!")

                            # 디버깅 로그 추가
                            print("After Generation and Integration - climate_data:")
                            print(st.session_state.climate_data.head(5))
                            print("After Generation and Integration - climate_data_integration:")
                            print(st.session_state.climate_data_integration.head(5))

                        except Exception as e:
                            st.error(f"Error during generation and integration: {e}")

                    # 통합된 버튼
                    if st.button("Generate and Send Climate Data to Integration"):
                        generate_and_send_climate_data()

                    # 데이터 다운로드 버튼
                    if st.session_state.climate_data is not None:
                        try:
                            csv = st.session_state.climate_data.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "Download Processed Climate Data",
                                data=csv,
                                file_name="processed_climate_data.csv",
                                mime="text/csv",
                                key="climate_data_download_btn"
                            )
                        except Exception as e:
                            st.error(f"Error during download: {e}")

        
        # ==========================
        # 1.3.  Economic Data Tab
        # ==========================
        with econ_col:
            st.subheader("Economic data")

            econ_file = st.file_uploader("Upload Economic CSV", type=["csv"], key="econ_file")  # 고유 key 설정

            if econ_file:
                try:
                    # 데이터 읽기
                    econ_data = pd.read_csv(econ_file, encoding="utf-8", parse_dates=["date"])
                    econ_data.columns = econ_data.columns.str.lower()  # 컬럼 이름을 소문자로 변환
                except UnicodeDecodeError:
                    econ_data = pd.read_csv(econ_file, encoding="cp949", parse_dates=["date"])
                    econ_data.columns = econ_data.columns.str.lower()  # 컬럼 이름을 소문자로 변환

                
                # 데이터 미리보기
                st.success("Economic Data Uploaded Successfully!")
                st.write("Economic Data Preview (First 10 Rows):")
                st.dataframe(econ_data.head(10))

                # 보간법 적용 버튼
                if st.button("Interpolate Economic Data"):
                    try:
                        # 중복 제거 및 날짜 기준 정렬
                        econ_data = econ_data.groupby("date").mean().reset_index()
                        econ_data.set_index("date", inplace=True)

                        # 보간법 적용
                        econ_data["pri"] = econ_data["pri"].interpolate(method="spline", order=2)  # 스플라인 보간법
                        econ_data["cpi"] = econ_data["cpi"].interpolate(method="linear")
                        econ_data["cpi_rice"] = econ_data["cpi_rice"].interpolate(method="linear")
                        econ_data["cpi_ramen"] = econ_data["cpi_ramen"].ewm(span=7, adjust=False).mean()  # 지수 평활법
                        econ_data["cpi_icecream"] = econ_data["cpi_icecream"].rolling(window=7, min_periods=1).mean()  # 이동 평균
                        econ_data["cpi_beer"] = econ_data["cpi_beer"].rolling(window=7, min_periods=1).mean()
                        econ_data["cpi_liquor"] = econ_data["cpi_liquor"].interpolate(method="spline", order=2)
                        econ_data["csi"] = econ_data["csi"].ewm(span=7, adjust=False).mean()
                        econ_data["bsi"] = econ_data["bsi"].interpolate(method="spline", order=2)
                        econ_data["interest rate"] = econ_data["interest rate"].interpolate(method="linear")
                        econ_data["expected inflation"] = econ_data["expected inflation"].ewm(span=7, adjust=False).mean()


                        # 일별 데이터 생성
                        econ_data = econ_data.resample("D").interpolate().reset_index()


                        # 세션에 데이터와 변수 정보 저장
                        st.session_state.econ_data = econ_data
                        st.session_state.econ_columns = list(econ_data.columns)  # 변수 이름 저장
                        st.success("Economic Data Processed and Interpolated!")
                        st.write("Processed Economic Data (First 10 Rows):")
                        st.dataframe(econ_data.head(10))
                        
                        # 다운로드 버튼
                        csv = econ_data.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="Download Interpolated Economic Data",
                            data=csv,
                            file_name="interpolated_economic_data.csv",
                            mime="text/csv"
                        )
  
                    except Exception as e:
                        st.error(f"Error while interpolating data: {e}")

                # 데이터 전송 버튼
                if st.button("Send to Data Integration and Template"):
                    if "econ_data" in st.session_state:
                        # 전처리된 데이터를 Data Integration과 Prediction Template로 전송
                        st.session_state.econ_data_integration = st.session_state.econ_data
                        st.session_state.econ_data_template = st.session_state.econ_data
                        st.success("Economic Data sent to Data Integration and Preparing a Prediction Template!")
                    else:
                        st.error("Please interpolate the Economic Data before sending!")
                

    # ==========================
    # 2nd 서브탭: Data Integration
    # ==========================

    with data_tabs[1]:
        st.subheader("Data integration (Train Data Preparation)")

        # 데이터 상태 표시
        st.markdown("### Data Status:")
        status_cols = st.columns(3)  # 데이터 상태를 3개 열로 구성
        with status_cols[0]:
            if "sales_data_integration" in st.session_state:
                st.write("● Sales Data: Data Received ✅")
                st.markdown("##### Sales Data Preview:")
                st.dataframe(st.session_state.sales_data_integration.head(10))
            else:
                st.write("● Sales Data: Data Not Received ❌")

        with status_cols[1]:
            if "climate_data_integration" in st.session_state:
                st.write("● Climate Data: Data Received ✅")
                st.markdown("##### Climate Data Preview:")
                st.dataframe(st.session_state.climate_data_integration.head(10))
            else:
                st.write("● Climate Data: Data Not Received ❌")

        with status_cols[2]:
            if "econ_data_integration" in st.session_state:
                st.write("● Economic Data: Data Received ✅")
                st.markdown("##### Economic Data Preview:")
                st.dataframe(st.session_state.econ_data_integration.head(10))
            else:
                st.write("● Economic Data: Data Not Received ❌")

        # 변수 선택 섹션
        st.markdown("### Variables for Integration:")
        selection_cols = st.columns(3)  # 변수 선택을 위한 3개 열 구성

        # Sales Data 변수 선택
        with selection_cols[0]:
            st.subheader("Select Sales Data Variables")
            select_all_sales = st.checkbox("Select All Sales Variables", key="select_all_sales")
            if select_all_sales and "sales_data_integration" in st.session_state:
                sales_vars = st.session_state.sales_data_integration.columns.tolist()
            elif "sales_data_integration" in st.session_state:
                sales_vars = st.multiselect(
                    "Select variables from Sales Data",
                    st.session_state.sales_data_integration.columns.tolist(),
                    key="sales_vars_multiselect"
                )
            else:
                sales_vars = []

        # Climate Data 변수 선택
        with selection_cols[1]:
            st.subheader("Select Climate Data Variables")
            select_all_climate = st.checkbox("Select All Climate Variables", key="select_all_climate")
            if select_all_climate and "climate_data_integration" in st.session_state:
                climate_vars = st.session_state.climate_data_integration.columns.tolist()
            elif "climate_data_integration" in st.session_state:
                climate_vars = st.multiselect(
                    "Select variables from Climate Data",
                    st.session_state.climate_data_integration.columns.tolist(),
                    key="climate_vars_multiselect"
                )
            else:
                climate_vars = []

        # Economic Data 변수 선택
        with selection_cols[2]:
            st.subheader("Select Economic Data Variables")
            select_all_econ = st.checkbox("Select All Economic Variables", key="select_all_econ")
            if select_all_econ and "econ_data_integration" in st.session_state:
                econ_vars = st.session_state.econ_data_integration.columns.tolist()
            elif "econ_data_integration" in st.session_state:
                econ_vars = st.multiselect(
                    "Select variables from Economic Data",
                    st.session_state.econ_data_integration.columns.tolist(),
                    key="econ_vars_multiselect"
                )
            else:
                econ_vars = []

        # 데이터 통합 버튼 및 처리
        if st.button("Generate Training Data"):
            if all(k in st.session_state for k in ["sales_data_integration", "climate_data_integration", "econ_data_integration"]):
                try:
                    # 병합 전 date 열 타입 통일
                    st.session_state.sales_data_integration['date'] = pd.to_datetime(st.session_state.sales_data_integration['date'], errors='coerce')
                    st.session_state.climate_data_integration['date'] = pd.to_datetime(st.session_state.climate_data_integration['date'], errors='coerce')
                    st.session_state.econ_data_integration['date'] = pd.to_datetime(st.session_state.econ_data_integration['date'], errors='coerce')

                    # 병합 기준 열 강제 추가
                    if 'date' not in sales_vars:
                        sales_vars.append('date')
                    if 'date' not in climate_vars:
                        climate_vars.append('date')
                    if 'date' not in econ_vars:
                        econ_vars.append('date')

                    # 병합 수행
                    combined_data = st.session_state.sales_data_integration[sales_vars]
                    combined_data = combined_data.merge(st.session_state.climate_data_integration[climate_vars], on="date", how="inner")
                    combined_data = combined_data.merge(st.session_state.econ_data_integration[econ_vars], on="date", how="inner")


                    st.write("Combined Training Data Preview:")
                    st.dataframe(combined_data.head(10))

                    # Training 데이터 통합 후 Prediction 데이터와 열 순서 동기화
                    st.session_state.train_data = combined_data  # 통합 데이터 업데이트
                    if "predict_data" in st.session_state and st.session_state.train_data is not None:
                        st.session_state.train_data, st.session_state.predict_data = align_columns(
                            st.session_state.train_data,
                            st.session_state.predict_data
                        )

                    # 데이터 다운로드 버튼
                    csv = combined_data.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("Download Combined Training Data", csv, "combined_Training data.csv", "text/csv")
                
                except KeyError as e:
                    st.error(f"KeyError: {e}. Make sure the selected variables include the 'date' column.")
                except Exception as e:
                    st.error(f"Error combining data: {e}")
            else:
                st.error("Please send all datasets to Data Integration before combining.")



    # ==========================
    # 3rd 서브탭: Preparing a Prediction Template
    # ==========================
    with data_tabs[2]:
        st.subheader("Generate Template")

        # SKU 리스트 표시
        st.subheader("Lists of SKUs")
        if "sku_list" in st.session_state and st.session_state["sku_list"]:
            st.write("List of SKUs identified from Sales Data:")
            st.dataframe(pd.DataFrame(st.session_state["sku_list"], columns=["SKU"]))  # SKU 리스트 표시
        else:
            st.warning("No SKUs identified. Please upload Sales Data in the Data Cleaning tab.")  # SKU가 없을 경우 경고

        # 날짜 입력
        start_date = st.date_input("Start Date", value=datetime(2025, 1, 1))
        end_date = st.date_input("End Date", value=datetime(2025, 12, 31))

        # 기후 데이터 템플릿 다운로드
        st.subheader("Download Climate Data Template")
        if st.button("Generate Climate Data Template"):
            try:
                # 날짜 범위 기반 템플릿 생성
                date_range = pd.date_range(start=start_date, end=end_date, freq="D")
                climate_template = pd.DataFrame({
                    "date": date_range,
                    "temp_avg": [None] * len(date_range),
                    "temp_min": [None] * len(date_range),
                    "temp_max": [None] * len(date_range),
                    "precip_mm": [None] * len(date_range),
                    "wind_avg": [None] * len(date_range),
                    "hum_avg": [None] * len(date_range),
                    "pres_avg": [None] * len(date_range),
                    "snow_depth_max": [None] * len(date_range),
                    "cloud_avg": [None] * len(date_range),
                    "grnd_temp_avg": [None] * len(date_range),
                })

                # 템플릿 다운로드 버튼 생성
                csv = climate_template.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="Download Climate Data Template",
                    data=csv,
                    file_name="climate_data_template.csv",
                    mime="text/csv",
                )
                st.success("Climate Data Template generated successfully!")
            except Exception as e:
                st.error(f"Error generating climate data template: {e}")

        # 기후 데이터 업로드
        st.subheader("Upload Climate Data")
        climate_file = st.file_uploader("Upload Climate Data CSV", type=["csv"], key="climate_upload")

        if climate_file:
            try:
                # 파일 읽기
                climate_data = pd.read_csv(climate_file)

                # 데이터 타입 변환 및 결측값 처리
                numeric_columns = [
                    "temp_avg", "temp_min", "temp_max", "precip_mm", "wind_avg",
                    "hum_avg", "pres_avg", "snow_depth_max", "cloud_avg", "grnd_temp_avg"
                ]
                for col in numeric_columns:
                    if col in climate_data.columns:
                        # NaN 값 처리: 평균으로 대체
                        climate_data[col].fillna(climate_data[col].mean(), inplace=True)

                # 파생 변수 생성 함수
                def generate_derived_variables(data):
                    data["temp_range"] = data["temp_max"] - data["temp_min"]
                    data["temp_perceived"] = (
                        13.12 + 0.6215 * data["temp_avg"]
                        - 11.37 * data["wind_avg"] ** 0.16
                        + 0.3965 * data["temp_avg"] * data["wind_avg"] ** 0.16
                    )
                    data["pressure_humidity_index"] = data["pres_avg"] / data["hum_avg"]
                    data["wind_cloud_interaction"] = data["wind_avg"] * data["cloud_avg"]
                    data["rain_indicator"] = (data["precip_mm"] > 0).astype(int)
                    data["rain_wind_interaction"] = data["precip_mm"] * data["wind_avg"]
                    data["rain_cloud_interaction"] = data["precip_mm"] * data["cloud_avg"]
                    data["discomfort_index"] = (
                        0.81 * data["temp_avg"]
                        + 0.01 * data["hum_avg"] * (0.99 * data["temp_avg"] - 14.3) + 46.3
                    )
                    data["snow_indicator"] = (data["snow_depth_max"] > 0).astype(int)
                    data["discomfort_temp_interaction"] = data["discomfort_index"] * data["temp_avg"]
                    data["apparent_humidity_interaction"] = data["temp_perceived"] * data["hum_avg"]
                    data["temp_precip_interaction"] = data["temp_avg"] * data["precip_mm"]
                    data["humidity_wind_interaction"] = data["hum_avg"] * data["wind_avg"]
                    data["temp_cloud_interaction"] = data["temp_avg"] * data["cloud_avg"]
                    data["wind_temp_range_interaction"] = data["wind_avg"] * data["temp_range"]
                    data["precip_discomfort_interaction"] = data["precip_mm"] * data["discomfort_index"]
                    data["pressure_temp_interaction"] = data["pres_avg"] * data["temp_avg"]
                    data["humidity_temp_range_interaction"] = data["hum_avg"] * data["temp_range"]
                    return data

                # 파생 변수 생성
                climate_data = generate_derived_variables(climate_data)

                # 모든 변수가 포함되었는지 확인
                expected_columns = [
                    "temp_avg", "temp_min", "temp_max", "precip_mm", "wind_avg", "hum_avg", "pres_avg",
                    "snow_depth_max", "cloud_avg", "grnd_temp_avg", "temp_range", "temp_perceived",
                    "pressure_humidity_index", "wind_cloud_interaction", "rain_indicator",
                    "rain_wind_interaction", "rain_cloud_interaction", "discomfort_index",
                    "snow_indicator", "discomfort_temp_interaction", "apparent_humidity_interaction",
                    "temp_precip_interaction", "humidity_wind_interaction", "temp_cloud_interaction",
                    "wind_temp_range_interaction", "precip_discomfort_interaction", "pressure_temp_interaction",
                    "humidity_temp_range_interaction"
                ]

                missing_columns = [col for col in expected_columns if col not in climate_data.columns]
                if missing_columns:
                    st.error(f"Missing columns: {', '.join(missing_columns)}")
                else:
                    st.success("All 28 variables generated successfully!")

                st.write("Processed Climate Data Preview:")
                st.dataframe(climate_data.head())

                # 경제 변수 병합 및 예상값 생성 함수
                def generate_forecast_values(variable, data):
                    if variable == "PRI":
                        return data["PRI"].interpolate(method="linear").iloc[-1]
                    elif variable == "BSI":
                        return data["BSI"].interpolate(method="spline", order=2).iloc[-1]
                    elif variable == "CPI":
                        return data["CPI"].iloc[-1]
                    elif variable in ["CPI_Rice", "CPI_Ramen"]:
                        return data[variable].rolling(window=7, min_periods=1).mean().iloc[-1]
                    elif variable in ["CPI_Icecream", "CPI_Beer", "CPI_Liquor"]:
                        return data[variable].ewm(span=7, adjust=False).mean().iloc[-1]
                    elif variable == "Interest rate":
                        return data["Interest rate"].interpolate(method="linear").iloc[-1]
                    elif variable == "Expected inflation":
                        return data["Expected inflation"].ewm(span=7, adjust=False).mean().iloc[-1]
                    else:
                        return data[variable].mean()

                # 테스트 데이터 생성
                if st.button("Generate Test Dataset"):
                    try:
                        if "sku_list" in st.session_state:
                            sku_list = st.session_state["sku_list"]
                            prediction_datasets = []

                            for sku in sku_list:
                                sku_data = climate_data.copy()
                                sku_data["SKU"] = sku
                                prediction_datasets.append(sku_data)

                            # SKU별 데이터 병합
                            combined_data = pd.concat(prediction_datasets)

                                # Test Dataset 생성 후 열 순서 동기화
                            if "train_data" in st.session_state:
                                st.session_state.train_data, combined_data = align_columns(
                                    st.session_state.train_data,
                                    combined_data
                                )    

                            # 경제 데이터 병합
                            if "econ_data_template" in st.session_state:
                                econ_data = st.session_state["econ_data_template"]
                                econ_data["date"] = pd.to_datetime(econ_data["date"], errors="coerce")
                                combined_data["date"] = pd.to_datetime(combined_data["date"], errors="coerce")

                                for column in econ_data.columns:
                                    if column != "date":
                                        combined_data[column] = generate_forecast_values(column, econ_data)

                            st.success("Test Dataset Generated!")
                            st.write("Generated Test Dataset Preview:")
                            st.dataframe(combined_data.head())

                            # 다운로드 버튼
                            csv = combined_data.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "Download Test Dataset",
                                csv,
                                "test_dataset.csv",
                                "text/csv"
                            )
                        else:
                            st.error("No SKUs identified. Please upload Sales Data in the Data Cleaning tab.")
                    except Exception as e:
                        st.error(f"Error processing data: {e}")

            except Exception as e:
                st.error(f"Error processing data: {e}")



    # ==========================
    # 4th 서브탭: Variables Details
    # ==========================
    with data_tabs[3]:
        st.header("Variable details")
        st.markdown(
            """
            This section provides detailed explanations for each variable in the dataset.

            ### Climate Variables: Definitions and Calculations

            #### Essential Climate Variables (ECVs)
            | Variable        | Formula                  |
            |---------------------------|--------------------------|
            | Average daily temperature (°C)   | temp_avg         |
            | Minimum daily temperature (°C)   | temp_min         |
            | Maximum daily temperature (°C)   | temp_max         |
            | Daily precipitation (mm)         | precip_mm        |
            | Average wind speed (m/s)         | wind_avg         |
            | Average daily relative humidity (%) | hum_avg        |
            | Average daily pressure (hPa)     | pres_avg         |
            | Maximum daily snow depth (cm)    | snow_depth_max   |
            | Average cloud coverage (1/10)    | cloud_avg        |
            | Average ground temperature (°C)  | grnd_temp_avg    |

            #### Derived Climate Variables
            | Variable               | Formula                                                                 |
            |---------------------------------|-------------------------------------------------------------------------|
            | Temperature range (°C)          | temp_range = temp_max - temp_min                                       |
            | Perceived temperature (°C)      | temp_perceived = 13.12 + 0.6215*temp_avg - 11.37*wind_avg^0.16 + 0.3965*temp_avg*wind_avg^0.16 |
            | Pressure to humidity ratio      | pressure_humidity_index = pres_avg / hum_avg                          |
            | Wind and cloud interaction      | wind_cloud_interaction = wind_avg * cloud_avg                         |
            | Rain indicator                  | rain_indicator = 1 (precip_mm > 0), 0 (precip_mm = 0)                 |
            | Rain and wind interaction       | rain_wind_interaction = precip_mm * wind_avg                          |
            | Rain and cloud interaction      | rain_cloud_interaction = precip_mm * cloud_avg                        |
            | Discomfort index                | discomfort_index = 0.81*temp_avg + 0.01*hum_avg*(0.99*temp_avg - 14.3) + 46.3 |
            | Discomfort and temperature interaction | discomfort_temp_interaction = discomfort_index * temp_avg       |
            | Perceived temperature and humidity interaction | apparent_humidity_interaction = temp_perceived * hum_avg |
            | Snow indicator                  | snow_indicator = 1 (snow_depth_max > 0), 0 (snow_depth_max = 0)       |

            #### Secondary Derived Variables
            | Variable                       | Formula                                      |
            |-----------------------------------------|---------------------------------------------|
            | Temperature and precipitation interaction | temp_precip_interaction = temp_avg * precip_mm |
            | Humidity and wind interaction           | humidity_wind_interaction = hum_avg * wind_avg |
            | Temperature and cloud interaction       | temp_cloud_interaction = temp_avg * cloud_avg |
            | Temperature range and wind interaction  | wind_temp_range_interaction = wind_avg * (temp_max - temp_min) |
            | Precipitation and discomfort interaction | precip_discomfort_interaction = precip_mm * discomfort_index |
            | Pressure and temperature interaction    | pressure_temp_interaction = pres_avg * temp_avg |
            | Humidity and temperature range interaction | humidity_temp_range_interaction = hum_avg * (temp_max - temp_min) |

            ### Economic Variables Definitions
            | Variable (Abbreviation)    | Definition                                                                                     |
            |----------------------------|-----------------------------------------------------------------------------------------------|
            | **PRI**                    | Past Year Inflation Perception: Indicates the perceived rate of inflation over the past year.  |
            | **CPI**                    | Consumer Price Index: Measures changes in the prices of goods and services purchased by consumers. |
            | **CPI_Rice**               | Consumer Price Index - Rice: Reflects changes in the cost of rice as part of CPI.             |
            | **CPI_Ramen**              | Consumer Price Index - Ramen: Reflects changes in the cost of ramen products.                 |
            | **CPI_Icecream**           | Consumer Price Index - Ice cream: Reflects changes in the cost of ice cream products.         |
            | **CPI_Beer**               | Consumer Price Index - Beer: Reflects changes in the cost of beer products.                   |
            | **CPI_Liquor**             | Consumer Price Index - Liquor: Reflects changes in the cost of liquor products.               |
            | **CSI**                    | Consumer Sentiment Index: Measures consumer confidence in economic conditions.                |
            | **BSI**                    | Business Sentiment Index: Reflects business confidence in the economic environment.           |
            | **Interest Rate**          | Central Bank Interest Rate: Represents the central bank's interest rate affecting borrowing costs. |
            | **Expected Inflation**     | Expected Inflation Rate: Reflects the anticipated rate of inflation over a specific time period. |
                

            ### Note on Economic Data Interpolation
            Interpolation methods vary depending on the characteristics of each variable and the type of data being interpolated (past or future). Below is a detailed breakdown:

            | **Variable**                          | **Past Data Interpolation**                                                                                     | **Future Data Interpolation**                                                                                  |
            |---------------------------------------|----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
            | **PRI (Past Year Inflation Perception)** | - **Linear Interpolation**: Suitable for gradual trends over time.<br> - Maintains smooth transitions in data. | - **Trend-based Forecasting**: Extrapolate using historical average growth rates or regression analysis.      |
            | **BSI (Business Sentiment Index)**     | - **Spline Interpolation**: Recommended for capturing smooth seasonal or business cycles.                      | - **Seasonality-based Forecasting**: Use patterns from past seasonal data (e.g., same month in prior years).  |
            | **CPI (Consumer Price Index)**         | - **Linear Interpolation**: Maintains consistent changes over time.                                             | - **Last Observation Carry Forward**: Use the most recent value for short-term stability.<br> - **Trend-based Extrapolation**: Apply linear trends for longer durations. |
            | **CPI_Rice**                           | - **Linear Interpolation**: Handles gaps in essential goods with stable demand.                                 | - **Recent Data Projection**: Short-term extrapolation with minimal adjustment for seasonal effects.          |
            | **CPI_Ramen**                          | - **Linear Interpolation**: Similar to Rice; effective for staple products.                                     | - **Exponential Smoothing**: Capture immediate changes while maintaining stable long-term trends.             |
            | **CPI_Icecream**                       | - **Linear Interpolation**: Handles gaps in seasonal product trends effectively.                                | - **Seasonality and Trend-based Forecasting**: Combine recent trends with seasonal adjustments.               |
            | **CPI_Beer**                           | - **Linear Interpolation**: Similar to Icecream; maintains consistency across gaps.                             | - **Recent Data with Smoothing**: Apply moving averages to smooth short-term fluctuations.                    |
            | **CPI_Liquor**                         | - **Spline Interpolation**: Suitable for capturing irregular or seasonal consumption patterns.<br> - Ensures smooth interpolation for large gaps. | - **Trend-based Forecasting**: Combine seasonal adjustments with recent growth rates for realistic projections. |
            | **CSI (Consumer Sentiment Index)**     | - **Spline Interpolation**: Smooths over sharp changes in consumer sentiment.                                   | - **Trend-based Extrapolation**: Uses historical patterns to project future consumer sentiment.               |
            | **Interest Rate**                      | - **Linear Interpolation**: Captures gradual changes in central bank policies.                                  | - **Economic Model Projection**: Uses macroeconomic models to forecast future rates.                          |
            | **Expected Inflation**                 | - **Exponential Smoothing**: Smooths fluctuations for more reliable predictions.                                | - **Recent Data Projection**: Projects short-term trends based on recent patterns.                            |
                    

            ### Note on Economic Data Interpolation

            #### Key Differences Between Past and Future Interpolation

            | **Aspect**                     | **Past Data Interpolation**                                                                              | **Future Data Interpolation**                                                                               |
            |--------------------------------|----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
            | **Purpose**                    | Filling gaps within the existing dataset.                                                                | Predicting values for periods beyond the available dataset.                                                |
            | **Techniques Used**            | Linear and spline interpolation to preserve data integrity.                                              | Regression, trend-based extrapolation, or smoothing techniques for realistic forecasts.                     |
            | **Assumptions**                | Assumes no external factors significantly alter the variable's trajectory.                               | Relies on historical trends, seasonality, and sometimes external variables to project future values.        |
            """
        )

        # Provide download link for explanations
        explanations = pd.DataFrame({
            "Variable Name (English)": [
                "temp_avg", "temp_min", "temp_max", "precip_mm", "wind_avg",
                "hum_avg", "pres_avg", "snow_depth_max", "cloud_avg", "grnd_temp_avg",
                "temp_range", "temp_perceived", "pressure_humidity_index", "wind_cloud_interaction",
                "rain_indicator", "rain_wind_interaction", "rain_cloud_interaction",
                "discomfort_index", "discomfort_temp_interaction", "apparent_humidity_interaction",
                "snow_indicator", "temp_precip_interaction", "humidity_wind_interaction",
                "temp_cloud_interaction", "wind_temp_range_interaction", "precip_discomfort_interaction",
                "pressure_temp_interaction", "humidity_temp_range_interaction"
            ],
            "Description": [
                "Average daily temperature", "Minimum daily temperature", "Maximum daily temperature",
                "Daily precipitation", "Average wind speed", "Average daily relative humidity",
                "Average daily pressure", "Maximum daily snow depth", "Average cloud coverage",
                "Average ground temperature", "Temperature range (max - min)",
                "Perceived temperature based on wind and temperature", "Pressure to humidity ratio",
                "Interaction between wind and cloud coverage", "Indicator for precipitation presence",
                "Interaction between precipitation and wind", "Interaction between precipitation and cloud coverage",
                "Discomfort index quantifying weather discomfort", "Interaction between discomfort index and temperature",
                "Interaction between perceived temperature and humidity", "Indicator for snowfall presence",
                "Interaction between temperature and precipitation", "Interaction between humidity and wind",
                "Interaction between temperature and cloud coverage", "Interaction between wind and temperature range",
                "Interaction between precipitation and discomfort index", "Interaction between pressure and temperature",
                "Interaction between humidity and temperature range"
            ],
            "Formula": [
                "temp_avg", "temp_min", "temp_max", "precip_mm", "wind_avg", "hum_avg",
                "pres_avg", "snow_depth_max", "cloud_avg", "grnd_temp_avg", "temp_max - temp_min",
                "13.12 + 0.6215*temp_avg - 11.37*wind_avg^0.16 + 0.3965*temp_avg*wind_avg^0.16",
                "pres_avg / hum_avg", "wind_avg * cloud_avg", "1 if precip_mm > 0 else 0",
                "precip_mm * wind_avg", "precip_mm * cloud_avg",
                "0.81*temp_avg + 0.01*hum_avg*(0.99*temp_avg - 14.3) + 46.3",
                "discomfort_index * temp_avg", "temp_perceived * hum_avg", "1 if snow_depth_max > 0 else 0",
                "temp_avg * precip_mm", "hum_avg * wind_avg", "temp_avg * cloud_avg",
                "wind_avg * (temp_max - temp_min)", "precip_mm * discomfort_index",
                "pres_avg * temp_avg", "hum_avg * (temp_max - temp_min)"
            ]
        })

        csv = explanations.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download Variable Explanations CSV",
            data=csv,
            file_name="variable_explanations.csv",
            mime="text/csv"
        )



# ==========================
# 탭2 Data Prediction : 상단 요약 영역
# ==========================
elif tab_selection == "Sales prediction":
    

    def train_and_predict_by_sku(train_data, predict_data, algorithm):
        results = []  # 각 SKU의 결과를 저장할 리스트
        sku_list = train_data['sku'].unique()  # SKU 리스트 추출

        for sku in sku_list:
            st.write(f"Processing SKU: {sku}")
            # SKU별 데이터 필터링
            train_sku_data = train_data[train_data['sku'] == sku]
            predict_sku_data = predict_data[predict_data['sku'] == sku]

            # train_and_predict 호출
            try:
                predict_sku_data, best_model, r2, mae, rmse, feature_importance = train_and_predict(
                    train_sku_data, predict_sku_data, algorithm
                )

                # 결과 저장
                results.append({
                    "SKU": sku,
                    "R²": r2,
                    "MAE": mae,
                    "RMSE": rmse,
                    "Feature Importance": feature_importance,
                    "Predictions": predict_sku_data
                })
            except Exception as e:
                st.warning(f"Error processing SKU {sku}: {e}")
                continue

        return results


    def dashboard_summary():
        """현재 상태와 선택 사항 요약"""
        st.markdown("### Dashboard Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Selected Algorithm:** {algorithm}")
        with col2:
            st.markdown(f"**Data Status:**")
            train_status = "Uploaded ✅" if st.session_state.train_data is not None else "Not Uploaded ❌"
            predict_status = "Uploaded ✅" if st.session_state.predict_data is not None else "Not Uploaded ❌"
            st.markdown(f"- Training Data: {train_status}")
            st.markdown(f"- Prediction Data: {predict_status}")

    # ==========================
    # 데이터 로드 및 초기화
    # ==========================
    if "train_data" not in st.session_state:
        st.session_state.train_data = None
    if "predict_data" not in st.session_state:
        st.session_state.predict_data = None
    if "sku_list" not in st.session_state:
        st.session_state.sku_list = []

    # ==========================
    # 사이드바 설정
    # ==========================

    st.sidebar.header("1. Select Algorithm:")
    algorithm = st.sidebar.selectbox(
        "Choose an Algorithm", ["Grid Search XGBoost", "Random Search XGBoost", "Bayesian Optimization XGBoost"]
    )

    st.sidebar.header("2. Upload Training Dataset:")
    train_file = st.sidebar.file_uploader("Upload Training CSV or Excel", type=["csv", "xlsx"], key="train")

    st.sidebar.header("3. Upload Prediction Dataset:")
    predict_file = st.sidebar.file_uploader("Upload Prediction CSV or Excel", type=["csv", "xlsx"], key="predict")
    
    # ==========================
    # 데이터 처리
    # ==========================

    if train_file:
        if train_file.name.endswith('.xlsx'):
            st.session_state.train_data = pd.read_excel(train_file, parse_dates=['date'])
        else:
            st.session_state.train_data = pd.read_csv(train_file, parse_dates=['date'])

        # 열 이름 통일 (소문자로 변환 및 공백 제거)
        st.session_state.train_data.columns = st.session_state.train_data.columns.str.lower().str.strip()

        # 데이터 타입 변환 강제 적용 (SKU 열 제외)
        for column in st.session_state.train_data.columns:
            if column != 'sku' and column != 'date':  # 'sku'와 'date' 열 제외
                st.session_state.train_data[column] = pd.to_numeric(st.session_state.train_data[column], errors='coerce')

        # SKU 데이터 처리
        if 'sku' in st.session_state.train_data.columns:
            # 데이터 타입 문자열로 변환
            st.session_state.train_data['sku'] = st.session_state.train_data['sku'].astype(str)

            # SKU 결측값 확인 및 처리
            missing_sku_count = st.session_state.train_data['sku'].isna().sum()
            if missing_sku_count > 0:
                st.warning(f"Training data contains {missing_sku_count} missing SKU values. Filling with 'Unknown'.")
                st.session_state.train_data['sku'].fillna('Unknown', inplace=True)

            # SKU 리스트 생성
            st.session_state.sku_list = st.session_state.train_data['sku'].unique().tolist()
            st.info(f"Training data contains {len(st.session_state.sku_list)} unique SKUs. Example: {st.session_state.sku_list[:5]}")

        st.sidebar.success("Training data loaded successfully!")

    if predict_file:
        if predict_file.name.endswith('.xlsx'):
            st.session_state.predict_data = pd.read_excel(predict_file, parse_dates=['date'])
        else:
            st.session_state.predict_data = pd.read_csv(predict_file, parse_dates=['date'])

        # 열 이름 통일 (소문자로 변환 및 공백 제거)
        st.session_state.predict_data.columns = st.session_state.predict_data.columns.str.lower().str.strip()

        # 데이터 타입 변환 강제 적용 (SKU 열 제외)
        for column in st.session_state.predict_data.columns:
            if column != 'sku' and column != 'date':  # 'sku'와 'date' 열 제외
                st.session_state.predict_data[column] = pd.to_numeric(st.session_state.predict_data[column], errors='coerce')

        # SKU 데이터 처리
        if 'sku' in st.session_state.predict_data.columns:
            # 데이터 타입 문자열로 변환
            st.session_state.predict_data['sku'] = st.session_state.predict_data['sku'].astype(str)

            # SKU 결측값 확인 및 처리
            missing_sku_count = st.session_state.predict_data['sku'].isna().sum()
            if missing_sku_count > 0:
                st.warning(f"Prediction data contains {missing_sku_count} missing SKU values. Filling with 'Unknown'.")
                st.session_state.predict_data['sku'].fillna('Unknown', inplace=True)

            # SKU 상태 점검
            unique_skus = st.session_state.predict_data['sku'].unique()
            st.info(f"Prediction data contains {len(unique_skus)} unique SKUs. Example: {unique_skus[:5]}")

        st.sidebar.success("Prediction data loaded successfully!")

    if train_file and predict_file:
        # 열 순서 동기화
        if st.session_state.train_data is not None and st.session_state.predict_data is not None:
            st.session_state.train_data, st.session_state.predict_data = align_columns(
                st.session_state.train_data,
                st.session_state.predict_data
            )

    # ==========================
    # 모델 학습 및 예측 함수
    # ==========================

    def train_and_predict(train_data, predict_data, algorithm):
        # 데이터 전처리: 컬럼 이름 소문자 변환 및 공백 제거
        train_data.columns = train_data.columns.str.strip().str.lower()
        predict_data.columns = predict_data.columns.str.strip().str.lower()

        # 변수 간 열 순서 동기화
        train_data, predict_data = align_columns(train_data, predict_data)

        # X_train, y_train, X_pred 생성
        if 'sales' not in train_data.columns:
            raise ValueError("'sales' column is missing in training data.")
        if 'date' not in train_data.columns or 'date' not in predict_data.columns:
            raise ValueError("'date' column is missing in one of the datasets.")

        X_train = train_data.drop(columns=['sales', 'date'], errors='ignore')
        y_train = train_data['sales']
        X_pred = predict_data.drop(columns=['date'], errors='ignore')

        # SKU 열을 범주형으로 변환 후 숫자로 인코딩
        if 'sku' in X_train.columns:
            X_train['sku'] = X_train['sku'].astype('category').cat.codes
            X_pred['sku'] = X_pred['sku'].astype('category').cat.codes

        # 데이터 타입 강제 변환 및 결측값 처리
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        y_train = pd.to_numeric(y_train, errors='coerce').fillna(0)
        X_pred = X_pred.apply(pd.to_numeric, errors='coerce').fillna(0)

        # 알고리즘 선택 및 모델 학습
        if algorithm == "Grid Search XGBoost":
            params = {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
            model = GridSearchCV(xgb.XGBRegressor(tree_method='hist', enable_categorical=True), params, cv=3)
        elif algorithm == "Random Search XGBoost":
            params = {'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
            model = RandomizedSearchCV(xgb.XGBRegressor(tree_method='hist', enable_categorical=True), params, n_iter=5, cv=3)
        else:
            model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=100, tree_method='hist', enable_categorical=True)

        # 모델 학습
        model.fit(X_train, y_train)
        best_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model

        # 예측 수행
        y_pred = best_model.predict(X_pred)
        predict_data['predicted_sales'] = y_pred

        # 평가 지표 계산
        r2 = r2_score(y_train, best_model.predict(X_train))
        mae = mean_absolute_error(y_train, best_model.predict(X_train))
        rmse = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))

        # Feature Importance 계산
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                "Feature": X_train.columns,
                "Importance (%)": best_model.feature_importances_ * 100
            }).sort_values(by="Importance (%)", ascending=False)

            # 컬럼 이름 공백 제거
            feature_importance.columns = feature_importance.columns.str.strip()
        else:
            feature_importance = pd.DataFrame()

        return predict_data, best_model, r2, mae, rmse, feature_importance




    # ==========================
    # 탭 구성
    # ==========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Preview", "Model Performance", "Predictions", "Algorithm Explanation", "Metrics Explanation"])

    # 탭 1: 데이터 미리 보기
    with tab1:
        st.title("Data Preview")
        dashboard_summary()
        st.subheader("Training Data")
        if st.session_state.train_data is not None:
            AgGrid(st.session_state.train_data)

        st.subheader("Prediction Data")
        if st.session_state.predict_data is not None:
            AgGrid(st.session_state.predict_data)
       
    

    # 탭 2: 모델 성능
    with tab2:
        st.title("Model Performance")
        dashboard_summary()
        st.markdown(
            """
            The **Model Performance** tab provides an overview of the model's evaluation metrics,
            such as R², MAE, and RMSE, alongside the importance of features used for predictions.
            The insights derived here directly impact the predictions displayed in the **Predictions** tab.
            """
        )

        # 상태 초기화
        if "predict_clicked" not in st.session_state:
            st.session_state.predict_clicked = False

        # Train and Predict 버튼
        if st.sidebar.button("Train and Predict"):
            if st.session_state.train_data is not None and st.session_state.predict_data is not None:
                # SKU별 분석 수행
                results = train_and_predict_by_sku(
                    st.session_state.train_data, st.session_state.predict_data, algorithm
                )

                # SKU별 결과 저장
                st.session_state.sku_results = results
                st.session_state.predict_clicked = True
                st.success("SKU-wise prediction completed!")

        # Train and Predict 이후 렌더링
        if st.session_state.predict_clicked:
            st.header("Overall Model Performance (SKU-wise)")

            # 전체 SKU 결과 테이블 생성
            all_sku_results = pd.DataFrame(
                [
                    {"SKU": result["SKU"], "R²": result["R²"], "MAE": result["MAE"], "RMSE": result["RMSE"]}
                    for result in st.session_state.sku_results
                ]
            )
            st.write("Summary of SKU-wise Results:")
            st.dataframe(all_sku_results)

            # CSV 다운로드 버튼
            csv_all_sku_results = all_sku_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download SKU-wise Results CSV",
                data=csv_all_sku_results,
                file_name="sku_wise_results.csv",
                mime="text/csv",
            )

            # SKU 선택 UI 추가
            sku_list = [result["SKU"] for result in st.session_state.sku_results]
            selected_sku = st.selectbox("Select an SKU to view detailed results:", sku_list)

            # 선택된 SKU의 결과 표시
            if selected_sku:
                for result in st.session_state.sku_results:
                    if result["SKU"] == selected_sku:
                        r2 = result["R²"]
                        mae = result["MAE"]
                        rmse = result["RMSE"]
                        feature_importance = result["Feature Importance"]
                        predicted_data = result["Predictions"]

                        st.subheader(f"Details for SKU: {selected_sku}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("R² Score", f"{r2:.2%}")
                        col2.metric("MAE", f"{mae:.2f}")
                        col3.metric("RMSE", f"{rmse:.2f}")

                        # Feature Importance 시각화
                        if not feature_importance.empty:  # 데이터가 비어 있지 않은 경우에만 실행
                            st.subheader(f"Feature Importance for SKU: {selected_sku}")
                            fig1 = px.bar(
                                feature_importance.head(10),  # 상위 10개만 표시
                                x="Importance (%)",
                                y="Feature",
                                orientation="h",
                                title=f"Feature Importance (Top 10) for SKU: {selected_sku}",
                                text_auto=".2f",
                                color="Importance (%)",
                                color_continuous_scale="Blues",
                            )
                            fig1.update_layout(yaxis=dict(categoryorder="total ascending"), height=400)
                            st.plotly_chart(fig1, use_container_width=True)
                        else:
                            st.warning(f"No feature importance data available for SKU: {selected_sku}")


                        # CSV 다운로드 버튼
                        csv_feature_importance = feature_importance.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"Download Feature Importance CSV for SKU: {selected_sku}",
                            data=csv_feature_importance,
                            file_name=f"feature_importance_{selected_sku}.csv",
                            mime="text/csv",
                        )

                        # 예측 데이터 다운로드
                        csv_predicted = predicted_data.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"Download Predicted Data for SKU: {selected_sku}",
                            data=csv_predicted,
                            file_name=f"predicted_data_{selected_sku}.csv",
                            mime="text/csv",
                        )

            # Time-based Feature Importance Analysis
            st.header("Time-based Feature Importance Analysis")
            st.markdown("#### Select Time Unit for Time-based Analysis:")
            time_unit = st.selectbox("Time Unit:", ["Daily", "Monthly", "Yearly"], index=1)

            # 선택된 SKU에 맞는 데이터 필터링
            if selected_sku:
                st.markdown(f"### Analyzing Time-based Feature Importance for SKU: {selected_sku}")
                train_data_filtered = st.session_state.train_data[
                    st.session_state.train_data["sku"] == selected_sku
                ]

                if train_data_filtered.empty:
                    st.warning(f"No data available for the selected SKU: {selected_sku}")
                else:
                    st.markdown("#### Select Features for Time-based Analysis:")

                    # Top 10 feature importance 선택
                    first_result = next((result for result in st.session_state.sku_results if result["SKU"] == selected_sku), None)
                    if first_result:
                        feature_importance_example = first_result["Feature Importance"]
                        top_10_features = feature_importance_example.nlargest(10, "Importance (%)")["Feature"].tolist()

                        selected_time_features = st.multiselect(
                            "Select Features for Time-based Analysis:",
                            feature_importance_example["Feature"].tolist(),
                            default=top_10_features,
                            key="time_based_feature_select",
                        )
                    else:
                        st.warning("No feature importance data available.")
                        selected_time_features = []

                    if selected_time_features:
                        try:
                            train_data_filtered["date"] = pd.to_datetime(train_data_filtered["date"])  # 날짜 형식 변환

                            # 시간 단위 설정
                            if time_unit == "Monthly":
                                train_data_filtered["time_unit"] = train_data_filtered["date"].dt.to_period("M").astype(str)
                            elif time_unit == "Yearly":
                                train_data_filtered["time_unit"] = train_data_filtered["date"].dt.to_period("Y").astype(str)
                            else:
                                train_data_filtered["time_unit"] = train_data_filtered["date"].dt.to_period("D").astype(str)

                            grouped_data = train_data_filtered.groupby("time_unit")
                            feature_importance_data = []

                            # 그룹별 Feature Importance 계산
                            for time, group in grouped_data:
                                if group.empty:
                                    st.warning(f"No data available for time unit: {time}")
                                    continue

                                X_train = group.drop(columns=["sales", "date", "time_unit"], errors="ignore")
                                y_train = group["sales"]

                                # SKU를 category로 변환
                                if "sku" in X_train.columns:
                                    X_train["sku"] = X_train["sku"].astype("category")

                                # XGBoost 모델 생성 및 학습
                                model = xgb.XGBRegressor(
                                    max_depth=5,
                                    learning_rate=0.1,
                                    n_estimators=100,
                                    enable_categorical=True
                                )
                                model.fit(X_train, y_train)

                                for feature, importance in zip(X_train.columns, model.feature_importances_ * 100):
                                    feature_importance_data.append(
                                        {"Time Unit": time, "Feature": feature, "Importance (%)": importance}
                                    )

                            feature_importance_df = pd.DataFrame(feature_importance_data)

                            # 데이터가 없으면 경고 표시
                            if feature_importance_df.empty:
                                st.warning("No feature importance data available for time-based analysis.")
                            else:
                                # 선택된 feature 필터링 및 시각화
                                filtered_importance_df = feature_importance_df[
                                    feature_importance_df["Feature"].isin(selected_time_features)
                                ]

                                if not filtered_importance_df.empty:
                                    fig2 = px.line(
                                        filtered_importance_df,
                                        x="Time Unit",
                                        y="Importance (%)",
                                        color="Feature",
                                        title=f"Time-based Feature Importance ({time_unit}) for SKU: {selected_sku}",
                                        markers=True,
                                    )
                                    st.plotly_chart(fig2, use_container_width=True)

                                    # CSV 다운로드 버튼
                                    csv2 = filtered_importance_df.to_csv(index=False).encode("utf-8")
                                    st.download_button(
                                        label="Download Time-based Feature Importance CSV",
                                        data=csv2,
                                        file_name=f"time_based_feature_importance_{time_unit.lower()}_{selected_sku}.csv",
                                        mime="text/csv",
                                    )
                                else:
                                    st.warning("Filtered data is empty. Please ensure selected features exist in the data.")

                        except Exception as e:
                            st.error(f"Error during time-based analysis: {e}")

    
    # 탭 3: 예측 결과
    with tab3:
        st.title("Predictions")
        dashboard_summary()  # 요약 정보 표시

        st.markdown("""
            The **Predictions** tab shows the predicted sales over time. This data is generated based on the
            model trained in the **Model Performance** tab and reflects the impact of key features.
        """)

        # 1. SKU 선택 UI
        if "sku_results" in st.session_state and st.session_state.sku_results:
            sku_list = [result["SKU"] for result in st.session_state.sku_results]  # SKU 리스트 생성
            selected_sku = st.selectbox("Select an SKU to view predictions:", sku_list)

            # 2. 선택된 SKU의 데이터 표시
            if selected_sku:
                # 선택된 SKU 데이터 필터링
                for result in st.session_state.sku_results:
                    if result["SKU"] == selected_sku:
                        predicted_data = result["Predictions"]
                        
                        # 선택된 SKU의 예측 데이터 테이블 표시
                        st.subheader(f"Predicted Sales for SKU: {selected_sku}")
                        st.write("Predicted Data:")
                        st.dataframe(predicted_data)

                        # 예측 데이터 시각화
                        st.subheader("Predicted Sales Over Time")
                        fig = px.line(
                            predicted_data,
                            x="date",
                            y="predicted_sales",
                            title=f"Predicted Sales Over Time for SKU: {selected_sku}",
                            markers=True,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # 예측 데이터 다운로드 버튼
                        csv_data = predicted_data[['date', 'sku', 'predicted_sales']].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                        st.download_button(
                            label=f"Download Predicted Data for SKU: {selected_sku}",
                            data=csv_data,
                            file_name=f"predicted_sales_{selected_sku}.csv",
                            mime="text/csv",
                        )
        else:
            st.warning("No predictions available. Please ensure the model is trained in the 'Model Performance' tab.")




    # Tab 4: Algorithm Explanation
    with tab4:
        st.title("Algorithm Explanation")
        
        # 1. Time Series Forecasting vs. Tree-based Algorithms
        st.markdown("### Time Series Forecasting vs. Tree-based Algorithms")
        st.markdown(
            """
            #### Time Series Forecasting (e.g., ARIMA, Exponential Smoothing)
            - **Advantages**:
            - Specifically designed for sequential data.
            - Captures trends and seasonality effectively.
            - Simple to interpret and implement.

            - **Limitations**:
            - Assumes linear relationships in data.
            - Requires stationarity or extensive preprocessing.
            - Less effective when multiple external variables influence predictions.

            #### Tree-based Algorithms (e.g., Random Forest, Gradient Boosting)
            - **Advantages**:
            - Handles complex, non-linear relationships.
            - Automatically selects important features.
            - Handles both numerical and categorical data.
            
            - **Limitations**:
            - Requires more computational power than traditional methods.
            - May overfit without proper tuning.
            """
        )

        # Visualization: ARIMA vs. XGBoost
        st.markdown("#### Visualization: ARIMA vs. XGBoost")

        # Generate synthetic data for ARIMA
        time = np.linspace(0, 10, 100)
        arima_trend = 10 + 2 * time
        arima_seasonal = 3 * np.sin(2 * np.pi * time / 5)
        arima_data = arima_trend + arima_seasonal

        # Generate synthetic data for XGBoost
        xgboost_time = np.linspace(0, 10, 100)
        xgboost_weather = 5 * np.sin(2 * np.pi * xgboost_time / 7)
        xgboost_event = np.where((xgboost_time > 3) & (xgboost_time < 6), 15, 0)
        xgboost_data = 10 + 1.5 * xgboost_time + xgboost_weather + xgboost_event

        # Plot ARIMA
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time, arima_data, label="Predicted Demand (ARIMA)", color="blue")
        plt.title("ARIMA: Simple Time Series Pattern Analysis")
        plt.xlabel("Time")
        plt.ylabel("Demand")
        plt.legend()
        plt.grid(True)

        # Plot XGBoost
        plt.subplot(2, 1, 2)
        plt.plot(xgboost_time, xgboost_data, label="Predicted Demand (XGBoost)", color="green")
        plt.title("XGBoost: Multivariate and Nonlinear Analysis")
        plt.xlabel("Time")
        plt.ylabel("Demand")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        st.pyplot(plt)

        st.markdown(
            """
            **Explanation**:
            - The top plot shows ARIMA's approach, which captures linear patterns such as trend and seasonality in time series data.
            - The bottom plot illustrates XGBoost's capability to model complex, nonlinear interactions between multiple variables like time, weather, and events.
            - XGBoost provides greater flexibility and accuracy when dealing with real-world data that includes external influences and non-linear relationships.
            """
        )

        # 2. Why XGBoost?
        st.markdown("### Why XGBoost?")
        st.markdown(
            """
            - **Comparison with Other Tree-based Models**:
            - **Random Forest**:
                - Ensemble of decision trees with bagging.
                - Good for reducing variance but slower in terms of prediction.
            - **Gradient Boosting Machines (GBMs)**:
                - Focuses on reducing bias by sequentially improving weak models.
                - Computationally intensive compared to XGBoost.
            - **CatBoost & LightGBM**:
                - CatBoost handles categorical features natively.
                - LightGBM is faster for large datasets but can be prone to overfitting.

            - **XGBoost's Strengths**:
            - Optimized for speed and performance.
            - Regularization techniques prevent overfitting.
            - Handles missing data automatically.
            - Parallel processing for faster training.
            """
        )

        # Visualization: XGBoost Mechanism
        st.markdown("#### Visualization: How XGBoost Works")
        from sklearn.tree import DecisionTreeRegressor
        x = np.linspace(0, 10, 50).reshape(-1, 1)
        y = 2 * x.ravel() + np.random.normal(0, 1, len(x))
        tree1 = DecisionTreeRegressor(max_depth=3).fit(x, y)
        residual1 = y - tree1.predict(x)
        tree2 = DecisionTreeRegressor(max_depth=3).fit(x, residual1)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.scatter(x, y, color="blue", label="Original Data")
        plt.plot(x, tree1.predict(x), color="green", label="Tree 1 Prediction")
        plt.title("Tree 1: Initial Prediction")
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.scatter(x, residual1, color="red", label="Residual (y - Tree 1)")
        plt.plot(x, tree2.predict(x), color="purple", label="Tree 2 Prediction")
        plt.title("Tree 2: Learning Residual")
        plt.legend()
        plt.subplot(1, 3, 3)
        final_prediction = tree1.predict(x) + tree2.predict(x)
        plt.scatter(x, y, color="blue", label="Original Data")
        plt.plot(x, final_prediction, color="black", label="Final Prediction")
        plt.title("Combined Prediction")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

        st.markdown(
            """
            **Explanation**:
            - The first tree predicts the initial approximation.
            - The second tree focuses on minimizing residuals (errors) from the first tree.
            - The final prediction combines both trees' outputs.
            """
        )

        # 3. Key Differences in Hyperparameter Optimization
        st.markdown("### Key Differences in Hyperparameter Optimization")
        st.markdown(
            """
            - **Grid Search XGBoost**:
            - Systematic approach to find the best hyperparameters.
            - Time-consuming but ensures optimal results.

            - **Random Search XGBoost**:
            - Randomly selects hyperparameters from a predefined range.
            - Faster than grid search but less exhaustive.

            - **Bayesian Optimization XGBoost**:
            - Uses probabilistic models to find optimal hyperparameters.
            - Balances exploration and exploitation efficiently.
            """
        )

        # Visualization: Hyperparameter Optimization
        st.markdown("#### Visualization: Hyperparameter Optimization")
        grid_x = np.linspace(0, 10, 5)
        grid_y = np.linspace(0, 10, 5)
        grid_points = [(x, y) for x in grid_x for y in grid_y]
        random_x = np.random.uniform(0, 10, 15)
        random_y = np.random.uniform(0, 10, 15)
        bayes_x = [2, 4, 6, 5, 5.5]
        bayes_y = [3, 5, 7, 6, 5.8]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Grid Search Visualization
        axes[0].scatter(*zip(*grid_points), color="blue", label="Grid Points")
        axes[0].set_title("Grid Search")
        axes[0].set_xlabel("Hyperparameter 1")
        axes[0].set_ylabel("Hyperparameter 2")
        axes[0].legend()

        # Random Search Visualization
        axes[1].scatter(random_x, random_y, color="orange", label="Random Points")
        axes[1].set_title("Random Search")
        axes[1].set_xlabel("Hyperparameter 1")
        axes[1].legend()

        # Bayesian Optimization Visualization
        axes[2].scatter(bayes_x, bayes_y, color="green", label="Bayesian Points")
        axes[2].plot(bayes_x, bayes_y, linestyle="--", color="green", alpha=0.5)
        axes[2].set_title("Bayesian Optimization")
        axes[2].set_xlabel("Hyperparameter 1")
        axes[2].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()




    with tab5:
        st.title("Metrics Explanation")
        
        # R² 설명
        st.markdown("### R² (R-squared)")
        st.markdown(
            """
            - **Definition**: R² measures how well the model explains the variance in the data. Its value ranges from 0 to 1 (or can be negative in rare cases).
            """
        )
        st.latex(r"R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}")
        st.markdown(
            """
            where:
            - \( SS_{\text{res}} \): Residual Sum of Squares.
            - \( SS_{\text{tot}} \): Total Sum of Squares.
            - **Characteristics**:
            - Closer to 1 indicates the model explains the data well.
            - Closer to 0 indicates poor explanatory power.
            - A negative value implies the model performs worse than using the mean as a prediction.
            """
        )
        
        # MAE 설명
        st.markdown("### MAE (Mean Absolute Error)")
        st.markdown(
            """
            - **Definition**: MAE is the average of the absolute differences between the predicted values and the actual values.
            """
        )
        st.latex(r"\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|")
        st.markdown(
            """
            where:
            - \( y_i \): Actual value.
            - \( \hat{y}_i \): Predicted value.
            - **Characteristics**:
            - A lower value indicates smaller average error.
            - Less sensitive to outliers.
            """
        )
        
        # RMSE 설명
        st.markdown("### RMSE (Root Mean Squared Error)")
        st.markdown(
            """
            - **Definition**: RMSE is the square root of the average squared differences between predicted values and actual values.
            """
        )
        st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}")
        st.markdown(
            """
            - **Characteristics**:
            - A lower value indicates smaller overall error.
            - More sensitive to larger errors, making it useful for emphasizing significant deviations.
            """
        )
        
        # Key Differences Table
        st.markdown(
            """
            ### Key Differences
            | Metric       | Definition                      | Characteristics                                |
            |--------------|----------------------------------|-----------------------------------------------|
            | **R²**       | Explains variance in data       | Closer to 1 indicates better explanatory power. |
            | **MAE**      | Mean of absolute errors         | Treats small and large errors equally.        |
            | **RMSE**     | Root mean squared error         | Penalizes large errors more heavily.          |
            
            ### When to Use
            - **R²**: Useful for assessing overall explanatory power of the model.
            - **MAE**: Good for understanding average errors in a straightforward manner.
            - **RMSE**: Helpful when emphasizing the importance of larger errors.
            """
        )

    # Handle cases where data is not loaded
    if st.session_state.train_data is None or st.session_state.predict_data is None:
        st.sidebar.warning("Please upload both training and prediction datasets to proceed.")

# 하단 Copyright 문구 추가
footer = """
    <style>
        footer {
            visibility: hidden;
        }
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #333;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 12px;
            font-family: 'Arial', sans-serif;
        }
    </style>
    <div class="footer">
        © 2025 Pulmuone. Frost. All Rights Reserved.
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)