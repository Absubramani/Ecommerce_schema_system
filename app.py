import streamlit as st
from inference import generate_schema

st.set_page_config(page_title="Ecommerce Schema Intelligence", layout="wide")

st.title("🛒 Ecommerce Schema Generator")

requirement = st.text_area(
    "Requirement",
    placeholder="Create ecommerce schema for tables customers, orders, payments",
    height=120
)

if st.button("Generate Schema"):

    if not requirement.strip():
        st.error("Enter requirement")
        st.stop()

    with st.spinner("Generating..."):
        schema, validation = generate_schema(requirement)

    st.code(schema, language="sql")

    if validation["is_valid"]:
        st.success("Schema valid ✅")
    else:
        st.warning("Validation issues:")
        for e in validation["errors"]:
            st.write("-", e)
