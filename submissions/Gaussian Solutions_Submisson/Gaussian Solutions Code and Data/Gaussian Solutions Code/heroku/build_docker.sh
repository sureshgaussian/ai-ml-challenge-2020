echo "Removing older flask code"
rm -rf ./flask_app 
echo "Copying flask code.."
cp -rf ../flask_app ./
echo "Removing older inference code"
rm -rf ./Inference
echo "Copying flask code.."
cp -rf ../Inference ./
echo "Removing older common code"
rm -rf ./common
echo "Copying flask code.."
cp -rf ../common ./
echo "building docker image..."
#docker build --tag eula:1.0 .
docker build --tag eula .


