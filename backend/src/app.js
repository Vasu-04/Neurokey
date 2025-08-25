const express = require("express")
const app = express()
const cors = require("cors")
const signUpRouter = require("./Router/SignUpRouter")

app.use(express.json())
app.use(cors())
app.use(express.urlencoded({ extended: true }))
app.use("/signUp",signUpRouter)
module.exports = app