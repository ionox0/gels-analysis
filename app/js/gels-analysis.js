// *******************
// Gel Image Functions
// *******************

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
          console.log ('The image size is '+origWidth+'*'+origHeight);
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
            img.attr('src', e.target.result);
            resizedWidth = img.width();
            resizedHeight = img.height();
        };
        reader.readAsDataURL(input.files[0]);





        // setRoiListeners();

        // Selection lib init
        $('#gel-image').imgAreaSelect({
            handles: true,
            onSelectEnd: endSelect
        });
    }
}

$("#the-file").change(function(){
    readURL(this);
});

var rois;
function endSelect(img, selection) {
    console.log(selection);

    var ratio = origWidth / resizedWidth;
    var original_x1 = Math.round(selection.x1 * ratio);
    var original_y1 = Math.round(selection.y1 * ratio);
    var original_x2 = Math.round(selection.x2 * ratio);
    var original_y2 = Math.round(selection.y2 * ratio);

    // Todo - support multiple ROIs
    rois = [];
    var roi = {
        'x_start': original_x1,
        'x_end': original_x2,
        'y_start': original_y1,
        'y_end': original_y2,
    };

    rois.push(roi);
}


function preview(img, selection) {
    var scaleX = 100 / (selection.width || 1);
    var scaleY = 100 / (selection.height || 1);

    $('#ferret + div > img').css({
        width: Math.round(scaleX * 400) + 'px',
        height: Math.round(scaleY * 300) + 'px',
        marginLeft: '-' + Math.round(scaleX * selection.x1) + 'px',
        marginTop: '-' + Math.round(scaleY * selection.y1) + 'px'
    });
}


function callRotateImage(){
    rotateBase64Image($('#gel-image').attr('src'));
}

function rotateBase64Image(base64data) {
    var canvas = document.createElement("canvas");
    canvas.setAttribute("id", "hidden-canvas");
    canvas.style.display = "none";
    document.body.appendChild(canvas);
    var ctx = canvas.getContext("2d");

    var image = new Image();
    image.src = base64data;
    image.onload = function() {
        var w = image.width;
        var h = image.height;
        var rads = 90 * Math.PI/180;
        var c = Math.cos(rads);
        var s = Math.sin(rads);
        if (s < 0) { s = -s; }
        if (c < 0) { c = -c; }
        //use translated width and height for new canvas
        canvas.width = h * s + w * c;
        canvas.height = h * c + w * s;
        //draw the rect in the center of the newly sized canvas
        ctx.translate(canvas.width/2, canvas.height/2);
        ctx.rotate(90 * Math.PI / 180);
        ctx.drawImage(image, -image.width/2, -image.height/2);

        $('#gel-image').attr('src', canvas.toDataURL());
        document.body.removeChild(canvas);
    };

}


// var rois = [];
// function setRoiListeners(){
//     var img_sel = $('#gel-image');
//     var roi = {};
//
//     img_sel.mousedown(function(e) {
//         var offset = $(this).offset();
//         roi.x_start = Math.floor(e.pageX - offset.left);
//         roi.y_start = Math.floor(e.pageY - offset.top);
//         event.preventDefault();
//         console.log("down", roi);
//     });
//
//     img_sel.mouseup(function(e) {
//         var offset = $(this).offset();
//         roi.x_end = Math.floor(e.pageX - offset.left);
//         roi.y_end = Math.floor(e.pageY - offset.top);
//         console.log("up", roi);
//         rois.push(roi);
//         roi = {};
//     });
// }

function analyze(){
    $('#gel-analysis-result').text('Analyzing...');

    var fileInput = document.getElementById('the-file');
    var file = fileInput.files[0];
    var formData = new FormData();

    colnames = $('#the-column-names').val();
    formData.append('file', file);
    formData.append('rois', JSON.stringify(rois));
    formData.append('min-y-value', $('#min-y-value').val());
    formData.append('max-y-value', $('#max-y-value').val());
    formData.append('threshold', $('#threshold').val());

    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {

            if (xhr.status === 200) {

                // convert to Base64
                var imgsrc = '/result';
                // var imgsrc = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(xhr.responseText)));
                var img = new Image(1, 1); // width, height values are optional params

                img.onload = function() {
                    $("#gel-analysis-result").html('<img src="' + img.src + '" />');
                    // $("#gel-analysis-result").html(img);
                    // document.write('<img src="' + img.src + '" height="100" width="200"/>');
                };

                img.src = imgsrc;
                // $('#gel-analysis-result').empty().html('<img src="data:image/png;base64,' + b64Response + '" />');

            } else {
                $('body').empty().append(xhr.responseText);
                console.log("Error", xhr.statusText);
            }

        }
    };
    xhr.open('POST', '/analyze', true);
    xhr.send(formData);
}