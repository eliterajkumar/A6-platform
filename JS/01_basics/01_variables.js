const accountId = 144553
let accountEmail = "rajkumar@google.com"
var accountPassword = "12345"
accountCity = "Ahemadabad"
let accountState;
//accountId = 2  // not allowed
accountEmail = "elite@hdfc.com"
accountPassword = "54321"
accountCity = "Noida"

console.log(accountId,);

/*
 prefer not to use var 
 because of issue in block scope and functional scope

*/



console.table([accountId, accountEmail, accountPassword, accountCity, accountState]);