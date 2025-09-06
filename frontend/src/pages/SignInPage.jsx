import React from 'react'
import axios from 'axios'
import { useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'

const SignInPage = () => {
  const Navigate = useNavigate();
  const [errorMessage, seterrorMessage] = useState("")
  const [formData, setformData] = useState({})
  const onEmailChange = (e) => {
    setformData({ ...formData, email: e.target.value })
  }
  const onPasswordChange = (e) => {
    setformData({ ...formData, password: e.target.value })
  }
  const handleSubmit = async (e) => {
    e.preventDefault();
    await axios.post("http://localhost:3000/signIn/", { formData })
      .then((res) => {
        const out = res.data.out
        if (out == true) {
          alert("Sign In Success")
          Navigate("/HomePage") // Replace with actual home page route
        }
        else {
          seterrorMessage("Email Or Password is incorrect")
        }
        console.log(res)
      })
      .catch((err) => {
        console.log(err)
      })
  }
  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="email" placeholder='Email' onChange={onEmailChange} required />
        <input type="password" placeholder='Password' onChange={onPasswordChange} required />
        <button type='submit'>Sign In</button>
      </form>
      <div>
        {errorMessage}
      </div>
    </div>

  )
}

export default SignInPage
