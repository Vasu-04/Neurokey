const express = require("express")
const router = express.Router()
const newUserModel = require("../Models/newUserModel")
const existingUserModel = require("../Models/existingUserModel")

router.post("/", async (req, res) => {
    const { dwellTime, flightTime, interKeyInterval } = req.body;
    const newUsers = await newUserModel.find()
    let matchFound = false;
    // newUsers.forEach(async element => {
    //     if (element.email === req.body.formData.email && element.password === req.body.formData.password) {
    //         //such user Exist
    //         const userId = element._id;
    //         const existingUsers = await existingUserModel.find();
    //         let existingValidUsers = []; //storing th ekeystroke data for the user
    //         existingUsers.forEach(existingUser => {
    //             if (existingUser.userId == userId) {
    //                 existingValidUsers.push(existingUser)
    //             }
    //         });
    //         console.log("userId : ", userId)
    //         console.log(existingUsers)
    //         console.log(existingValidUsers)
    //         console.log("existingValidUsers length : ", existingValidUsers.length)
    //         if (existingValidUsers.length == 0) {//whether keystroke data exists for the user
    //             matchFound = true;
    //             //give access to user and add his keystroke data to existingUserModel

    //             const newExistingUser = new existingUserModel({
    //                 userId: userId,
    //                 dwellTime: dwellTime,
    //                 flightTime: flightTime,
    //                 interKeyTime: interKeyInterval
    //             })
    //             await newExistingUser.save()
    //             //console.log("New Existing User Added")
    //         }
    //         else {
    //             // ML Model Code Here
    //             // adding keystroke biometrics data to existingUserModel
    //         }
    //         if (matchFound) return res.status(200).json({ message: "Sign In Success", out: matchFound })

    //     }
    // });
    for (const element of newUsers) {
        if (element.email === req.body.formData.email && element.password === req.body.formData.password) {
            const userId = element._id;
            const existingUsers = await existingUserModel.find();
            const existingValidUsers = existingUsers.filter(u => u.userId == userId);

            if (existingValidUsers.length === 0) {
                const newExistingUser = new existingUserModel({
                    userId,
                    dwellTime: dwellTime,
                    flightTime: flightTime,
                    interKeyTime: interKeyInterval  // also fix field name here
                });
                await newExistingUser.save();
                return res.status(200).json({ message: "Sign In Success", out: true });
            } else {
                // ML Model Code Here
            }
        }
    }
    return res.status(200).json({ message: "Invalid Credentials", out: false });

    // if (!matchFound) {
    //     return res.status(200).json({ message: "Invalid Credentials", out: matchFound })
    // }
    // console.log("Users stored : ",newUsers)
    // console.log(req.body)
    // res.status(200).json({message : "data recieved success"})
})

module.exports = router