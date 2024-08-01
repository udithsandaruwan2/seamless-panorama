const selectImage = document.querySelector('.select-image');
const inputFile = document.querySelector('#file');
const imgArea = document.querySelector('#imgArea');

selectImage.addEventListener('click', function () {
    inputFile.click();
});

inputFile.addEventListener('change', function () {
    const files = this.files;

    if (files.length > 5) {
        alert("You can only upload up to 5 images.");
        return;
    }

    let totalSize = 0;
    const imageUrls = [];
    const validFiles = [];

    Array.from(files).forEach(file => {
        if (file.size > 2000000) { 
            alert(`Image ${file.name} is more than 2MB.`);
            return;
        }
        totalSize += file.size;
        validFiles.push(file);
    });

    if (totalSize > 5000000) { 
        alert("Total size of all images must be smaller than 5MB.");
        return;
    }

    validFiles.forEach(file => {
        const reader = new FileReader();
        reader.onload = () => {
            const imgUrl = reader.result;
            imageUrls.push(imgUrl);
            const imgContainer = document.createElement('div');
            imgContainer.classList.add('image-container');
            const img = document.createElement('img');
            img.src = imgUrl;
            const imgName = document.createElement('div');
            imgName.classList.add('image-name');
            imgName.textContent = file.name;
            imgContainer.appendChild(img);
            imgContainer.appendChild(imgName);
            imgArea.appendChild(imgContainer);

            if (imageUrls.length === validFiles.length) {
                imgArea.classList.add('active');
                localStorage.setItem('uploadedImages', JSON.stringify(imageUrls));
                window.location.href = '/preview';
            }
        };
        reader.readAsDataURL(file);
    });
});
