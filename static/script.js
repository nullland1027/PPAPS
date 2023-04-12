function displayFileName() {
    const fileInput = document.getElementById('file');
    const fileName = fileInput.files[0].name;
    const fileNameDisplay = document.getElementById('fileName');
    fileNameDisplay.innerHTML = `File selected：${fileName}`;
}

function checkFileSelected() {
    const fileInput = document.querySelector('input[type=file]');
    if (!fileInput.value) {
        alert('Please select a file to upload.');
        return false;
    }
    return true;
}

function validateForm() {
    let i;
    const radio_kind = document.getElementsByName("kind");
    const radio_al = document.getElementsByName("al");
    let formValid_kind = false;
    let formValid_al = false;
    for (i = 0; i < radio_kind.length; i++) {
        if (radio_kind[i].checked) {
            formValid_kind = true;
            break;
        }
    }
    for (i = 0; i < radio_al.length; i++) {
        if (radio_al[i].checked) {
            formValid_al = true;
            break;
        }
    }

    if (!formValid_kind || !formValid_al) {
        alert("Please select an option!");
        return false;
    }
    return true;
}

function test() {
    // 获取按钮元素
    const myButton = document.getElementById('myButton');

    // 监听按钮点击事件
    myButton.addEventListener('click', function() {
        // 使用 AJAX 或 Fetch API 发送请求到 Flask 后台
        fetch('/my-endpoint')
            .then(response => response.json())
            .then(data => console.log(data))
            .catch(error => console.error(error));
    });
}

function validatePassword() {
  var password = document.getElementsByName("password")[0].value;
  var confirmPassword = document.getElementsByName("confirm-password")[0].value;

  if (password != confirmPassword) {
    document.getElementsByName("password")[0].classList.add("border-danger");
    document.getElementsByName("confirm-password")[0].classList.add("border-danger");
    return false;
  } else {
    document.getElementsByName("password")[0].classList.remove("border-danger");
    document.getElementsByName("confirm-password")[0].classList.remove("border-danger");
    return true;
  }
}

