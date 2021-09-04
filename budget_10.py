import streamlit as st
import pandas as pd
import altair as alt

from urllib.error import URLError

@st.cache
def get_data():
    df = pd.read_excel("2010_2020_факт.xlsx")
    return df[df["Отчет"]=="Доходы"][df.columns[2:]].set_index("Наименование")

try:
    df = get_data()
    pos = st.multiselect(
        "Выберите статьи", list(df.index), list(df.index)[:5]
    )
    if not pos:
        st.error("Выберите хотя бы одну статью")
    else:
        data = df.loc[pos]
        data /= 1000000.0
        st.write("### Статьи доходов, млн руб., ", data.sort_index())

        data = data.T.reset_index()
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "Годы", "value": "Доходы, млн руб."}
        )
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="Годы:T",
                y=alt.Y("Доходы, млн руб.:Q", stack=None),
                color="Наименование:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)
except Exception as e:
    st.error(
        """
        Ошибка загрузки: %s
    """
        % e.reason
    )