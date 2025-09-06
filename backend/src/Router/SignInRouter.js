const express = require("express")
const router = express.Router()
const newUserModel = require("../Models/newUserModel")
const existingUserModel = require("../Models/existingUserModel")

router.post("/", async (req, res) => {
    const newUsers = await newUserModel.find()
    let matchFound = false;
    newUsers.forEach(async element => {
        if (element.email === req.body.formData.email && element.password === req.body.formData.password) {

            const userId = element._id;
            const existingUsers = await existingUserModel.find();
            let existingValidUsers = [];
            existingUsers.forEach(existingUser => {
                if (existingUser.userId === userId) {
                    existingValidUsers.push(existingUser)
                }
            });
            console.log(existingUsers)
            console.log(existingValidUsers)
            if (existingValidUsers.length == 0) {
                matchFound = true;
                // adding keystroke biometrics data to existingUserModel
                // const {dwellTime,flightTime,interKeyTime} = req.body.formData
                // const newExistingUser = new existingUserModel({
                //     userId : userId,
                //     dwellTime : dwellTime,
                //     flightTime : flightTime,
                //     interKeyTime : interKeyTime
                // })
                // await newExistingUser.save()
                // console.log("New Existing User Added")
            }
            else {
                // ML Model Code Here
                // adding keystroke biometrics data to existingUserModel
            }
            if (matchFound) return res.status(200).json({ message: "Sign In Success", out: matchFound })
        }
    });
    if (!matchFound) {
        return res.status(200).json({ message: "Invalid Credentials", out: matchFound })
    }
    // console.log("Users stored : ",newUsers)
    // console.log(req.body)
    // res.status(200).json({message : "data recieved success"})
})

module.exports = router