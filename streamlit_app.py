import streamlit as st

st.title("Hello World! 👋")
st.write("This is my very first app!")

# Ask for name
name = st.text_input("What is your name?")
if name:
    st.write(f"Hello {name}! Welcome to my app! 😊")

# Ask for age
age = st.slider("How old are you?", 0, 100, 25)
st.write(f"You are {age} years old.")

# Show balloons when clicked
if st.button("Click me for balloons! 🎈"):
    st.balloons()
