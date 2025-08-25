import React from 'react'

import { Routes, Route } from 'react-router-dom'
import SignUpPage from './pages/SignUpPage'
import SignInPage from './pages/signInPage'
const App = () => {
  return (
    <div className='mainBody'>
      <Routes>
        <Route path="/" element={<SignUpPage/>}/>
        <Route path="/signInPage" element={<SignInPage/>}/>
      </Routes>
    </div>
  )
}

export default App
