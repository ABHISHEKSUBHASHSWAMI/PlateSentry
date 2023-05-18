let arrow = document.querySelectorAll(".arrow");
for (var i = 0; i < arrow.length; i++) {
  arrow[i].addEventListener("click", (e)=>{
 let arrowParent = e.target.parentElement.parentElement;//selecting main parent of arrow
 arrowParent.classList.toggle("showMenu");
  });
}

let sidebar = document.querySelector(".sidebar");
let sidebarBtn = document.querySelector(".bx-vector");
console.log(sidebarBtn);
sidebarBtn.addEventListener("click", ()=>{
  sidebar.classList.toggle("close");
});

/*function loadImage(event) {
  var reader = new FileReader();
  var file = event.target.files[0];
  var ext = file.name.substring(file.name.lastIndexOf('.') + 1).toLowerCase();
  if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "gif") {
    reader.onload = function(){
      var uploadedImage = document.getElementById('uploaded-image');
      uploadedImage.src = reader.result;
    }
    reader.readAsDataURL(file);
  } else {
    alert("Only JPG, JPEG, PNG, and GIF files are allowed!");
  }
}


function loadVideo(event) {
  var reader = new FileReader();
  var file = event.target.files[0];
  var ext = file.name.substring(file.name.lastIndexOf('.') + 1).toLowerCase();
  if (ext == "mp4" || ext == "mkv" || ext == "avi" || ext == "3gp" || ext == "m4a" || ext == "mov") {
    reader.onload = function() {
      var uploadedVideo = document.getElementById('uploaded-video');
      uploadedVideo.src = reader.result;
      uploadedVideo.style.display = 'block'; // Show the video element
    };
    reader.readAsDataURL(file);
  } else {
    alert("Only MP4, MKV, AVI, and 3GP files are allowed!");
  }
}

*/


var liveVideo = document.getElementById('live-video');

if (liveVideo) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function(stream) {
      liveVideo.srcObject = stream;
    })
    .catch(function(error) {
      console.error('Error accessing the camera: ', error);
    });
}
