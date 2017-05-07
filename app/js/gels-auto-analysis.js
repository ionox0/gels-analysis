var cropper;
function initCropper(image){
    cropper = new Cropper(image);
}

var origHeight;
var origWidth;
var resizedHeight;
var resizedWidth;
function readURL(input) {
    if (input.files && input.files[0]) {

        // Set original height / width
        var newImg = new Image();

        newImg.onload = function() {
          origHeight = newImg.height;
          origWidth = newImg.width;
        };

        var reader1 = new FileReader();
        reader1.onload = function (e) {
            newImg.src = e.target.result;
        };
        reader1.readAsDataURL(input.files[0]);

        // Display resized image
        var reader = new FileReader();

        reader.onload = function (e) {
            var img = $('#gel-image');

            img.on('load', function(){
                resizedWidth = img.width();
                resizedHeight = img.height();
                console.log(resizedWidth, resizedHeight);

                initCropper(document.getElementById('gel-image'));
            });

            img.attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);

    }
}

$("#the-file").change(function(){
    readURL(this);
});


// Add ROI
var rois = [];
var labels = [];

$(function(){
    document.onkeypress = function (e) {
        e = e || window.event;

        if (e.keyCode != 49 && e.keyCode != 50) {
            return;
        }

        var crop_data = cropper.getData(true);
        var roi = {
            'x_start': crop_data.x,
            'x_end': crop_data.x + crop_data.width,
            'y_start': crop_data.y,
            'y_end': crop_data.y + crop_data.height
        };
        rois.push(roi);

        if (e.keyCode == 49) {
            labels.push('CTRL');
        } else if (e.keyCode == 50) {
            labels.push('DZ');
        }


        var naturalWidth = cropper.getImageData().naturalWidth;
        var width = cropper.getImageData().width;
        var ratio = width / naturalWidth;
        var moveAmount = - crop_data.width * ratio;

        // Move the crop region to next lane
        cropper.move(moveAmount, 0);
    };
});


// Upload Training Data
function uploadTrainingData(){
    $('#upload-result').text('Uploading...');

    // Set crop to full image
    // cropper.reset();
    cropper.zoomTo(0.1);
    cropper.moveTo(0);
    cropper.setCropBoxData(cropper.getCanvasData());
    cropper.getCroppedCanvas().toBlob(a);

    function a (blob) {
        var fileInput = document.getElementById('the-file');
        var file = fileInput.files[0];

        var formData = new FormData();
        formData.append('filename', file.name);
        formData.append('picture', blob);
        formData.append('rois', JSON.stringify(rois));
        formData.append('labels', JSON.stringify(labels));

        var xhr = new XMLHttpRequest();
        xhr.onreadystatechange = function () {
            if (xhr.readyState == XMLHttpRequest.DONE) {

                if (xhr.status === 200) {

                    $('#upload-result').text('Upload Complete');

                } else {
                    $('body').empty().append(xhr.responseText);
                    console.log("Error", xhr.statusText);
                }

            }
        };
        xhr.open('POST', '/upload_train_data', true);
        xhr.send(formData);

    }
}
