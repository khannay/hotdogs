//import * as tf from '@tensorflow/tfjs';


var wait = ms => new Promise((r, j)=>setTimeout(r, ms));


async function main() {
    const model = await tf.loadLayersModel('./model/model.json');
    document.getElementById('image_upload').onchange = function(ev) {
        var f = ev.target.files[0];
        var fr = new FileReader();
        var makePrediction = async function(img) {
            // We need to ensure that the image is actually loaded before we proceed.
            while(!img.complete) {
                await wait(100);
            }


	    let offset=tf.scalar(127.5);
            var tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]).toFloat().sub(offset).div(offset).expandDims();

            const prediction = model.predict(tensor);

	    
            var data = prediction.dataSync();
	    //alert(data[0]);
            document.getElementById('result').innerHTML = data[0] > 0.5 ? "Ruling: Not Hotdog" : "Ruling: Hotdog!";
        }
        var fileReadComplete = function(ev2) {
            document.getElementById('image').src = ev2.target.result;
            var img = new Image();
            img.src = ev2.target.result;
            makePrediction(img);
        };
        fr.onload = fileReadComplete;
                
        fr.readAsDataURL(f);
    }
}
main();
