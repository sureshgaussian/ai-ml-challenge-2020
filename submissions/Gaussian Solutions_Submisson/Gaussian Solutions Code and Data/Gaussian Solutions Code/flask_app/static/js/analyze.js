// Get a reference to the file input element & input label 

// Function to update the input placeholder
function input_filename() {
    var doc_element = document.getElementById("document");
    var document_label = document.getElementById("document_label");
  
    document_label.innerText = doc_element.files[0].name;
    console.log("In inputfile name function")
  }

 function flash_wait_message() {
  $("#message").text(' Analyzing document.. this may take a little while..')

 }

function getElementsByIdStartsWith(container, selectorTag, prefix) {
  var items = [];
  var myPosts = document.getElementById(container).getElementsByTagName(selectorTag);
  for (var i = 0; i < myPosts.length; i++) {
      //omitting undefined null check for brevity
      if (myPosts[i].id.lastIndexOf(prefix, 0) === 0) {
          items.push(myPosts[i]);
      }
  }
  return items;
}
function update_decisions( url ) {
  console.log("In update function ")

  //var buttons = dbuttons.querySelectorAll('[id^="inlineRadio_d"]');
  var buttons = document.querySelectorAll('[id^="inlineRadio_d"]');

  var decisions = []
  console.log("buttons:", buttons)
  for (let button of buttons) {
    console.log(button.checked)
    decisions.push(button.checked)
  }

  // send decisions to the server
  // Create a XMLHTTPRequest instance
  var request = new XMLHttpRequest();

  // Set the response type
  request.responseType = "json";

  // open request
  data = JSON.stringify(decisions)
  // Open and send the request
  request.open("post", url);
  request.send(data);
  // when we are here.. request.responseTex... has the html

  //$("#message").text(' Thank you for submitting your decisions')
  //window.location.href = 
  //$("#analysis_results").attr("style","display:none");
  //$("#analysis_results").hide()
  window.alert('Thank you for submitting your reviewed decisions. Your inputs will be used to improve the AI model\n\n We will take you to the home page')
  window.location.href = "/"

}