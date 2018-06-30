'use strict';

var app = angular.module('faceEmotion', ['ngMessages']);

app.controller('useWebCam', useWebCam)
app.controller('results', results)

var loadedPhoto;
var validfile;

angular.
module('faceEmotion').
filter('emoji', function() {
  return function(input) {

    var faces = {
      neutral: 'static/img/emoji/static/neutral.svg',
      angry: 'static/img/emoji/static/Angry_flat.png',
      sad: 'static/img/emoji/static/Cry_flat.png',
      happy: 'static/img/emoji/static/Lol_flat.png',
      surprise: 'static/img/emoji/static/Wow_flat.png',
      fear: 'static/img/emoji/static/fear.svg',
	  disgust: 'static/img/emoji/static/Disgust.jpg'
    }
    return faces[input]
  };
});



angular.
module('faceEmotion').
filter('percent', function() {
  return function(input) {
    input *= 100;
    var pct = input.toFixed(1);
    if (pct.charAt(pct.length - 1) === '0') {
      pct = pct.slice(0, pct.length - 2)
    }
    return pct;
  };
});

//  webcam foto, annulla, previsione

function useWebCam($http, $rootScope) {
  var w = this;
  var rt = $rootScope;

  w.showVideo = true;
  w.showCanvas = false;

  var video = document.querySelector('video');
  var canvas = window.canvas = document.querySelector('canvas');
  canvas.width = 480;
  canvas.height = 360;

  var constraints = {
    audio: false,
    video: true
  };

  w.snapPhoto = function() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').
    drawImage(video, 0, 0, canvas.width, canvas.height);

    w.showVideo = false;
    w.showCanvas = true;

    console.log('snap photo');
    w.snapped = canvas.toDataURL('image/jpeg;base64;', 0.1)

  }


w.submit = function() {
    console.log('submit photo');
    window.stream.getVideoTracks()[0].stop();
    // play waiting animation by setting some varianle WAITING to true
    rt.waiting = true;
    w.showCanvas = false;
    $http({
      method: 'POST',
      //url: 'http://54.227.229.33:5000/v1.0.0/predict',
      //location.href='/predict';
	  url: '/predict',
	  //url: 'http://127.0.0.1:5000/predict',
      params: {
        image_base64: w.snapped,
        annotate_image: true,
        crop_image: true
      }
    }).then(function success(data) {
      console.log('submit success......');
      //console.log(data.data);
      //console.log(w.snapped);
      //console.log(rt.results_received);
      //console.log(queryCommandEnabled);
      rt.waiting = false;
      rt.useWebCam = false;
      rt.results_received = true;
      console.log('resulttt______');
      rt.results = data.data
      rt.original = w.snapped
    }, function fail(data) {
      // set WAITING to false
      console.log('error: ', data);
    })
  }


  w.retake = function() {
    console.log('retaking photo');
    w.showVideo = true;
    w.showCanvas = false;
  }

//---------------------------------------


  function handleSuccess(stream) {
    window.stream = stream; // make stream available to browser console
    video.srcObject = stream;
  }

  function handleError(error) {
    console.log('navigator.getUserMedia error: ', error);
  }

  navigator.mediaDevices.getUserMedia(constraints).
  then(handleSuccess).catch(handleError);
}


//-------------result

function results($rootScope, $http) {
  var res = this;
  var rt = $rootScope;
  console.log('result...di result......');

  res.feedback = {};
  res.emote = {};
  res.submitted = {};
  //res.nofaces = rt.results.faces.length === 0;
  res.hideImg = false;
  res.hideCanvas = true;

  var img_container = document.querySelector('#imgcontainer');
  var img = document.querySelector('#og');
  var canvas = document.querySelector('#canvas');
  var rageFaces = [];
  console.log("img");
  console.log(img);

  img.addEventListener('load', function() {
    img_container.style.width = img.width + 'px';
    img_container.style.height = img.height + 'px';
    console.log("listenerrr");

  })
}

