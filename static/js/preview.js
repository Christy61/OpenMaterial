datas = null;
datas_olat = null;
fetch('data.json')
    .then(response => response.json())
    .then(data => {
        // const table = document.getElementById('myTable');
        const table = createTable(data["obj_list"], "myTable", 'obj');
        // tableContainer.appendChild(table);
        datas = data;
    })
    .catch(error => console.error(error));

function createTable(data, table_id, html_name) {
    const table = document.getElementById(table_id);
    // Create table rows
    data.forEach(item => {
        const row = document.createElement('tr');
        const objectId = item['id'];
        Object.values(item).forEach((value, index) => {
            if (index === 1) {
                const td = document.createElement('td');
                const a = document.createElement('a');
                a.href = `${html_name}.html?id=` + objectId.toString();
                a.textContent = value;
                td.appendChild(a);
                row.appendChild(td);
            } else if (index !== 2) {
                const td = document.createElement('td');
                td.textContent = value;
                row.appendChild(td);
            }
        });
        row.addEventListener('mouseover', showPreview);
        row.addEventListener('mouseout', hidePreview);
        table.appendChild(row);
    });

    return table;
}

function showPreview(event) {
    var previewImage = document.getElementById('preview-image');
    var currentRow = event.currentTarget;
    tableId = currentRow.parentNode.id;
    var rowData = currentRow.innerText.split("\t")[0];

    if (tableId === "myTable") {
        root = "files/lighting_patterns";
        lightIndex = "013";
        filename = "CA2.jpg";
        width = 100;
        height = 150;
        thisdata = datas['obj_list'][rowData - 1];
    } else {
        root = "files/OLAT";
        lightIndex = "000";
        filename = "A1.jpg"
        width = 100;
        height = 136;
        thisdata = datas_olat['obj_list'][rowData - 1];
    }

    // Position the popup next to the mouse pointer
    var rect = currentRow.getBoundingClientRect();
    var scrollTop = window.scrollY || document.documentElement.scrollTop;
    var scrollLeft = window.scrollX || document.documentElement.scrollLeft;
    // var x = rect.left + scrollLeft;
    var x = event.clientX;
    var y = rect.top + scrollTop;

    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');

    var thumbnailImage = new Image();
    thumbnailImage.src = `${root}/${thisdata['data_name']}/Lights/${lightIndex}/origin_smaller_thumbnail/${filename}`;
    thumbnailImage.onload = function () {
        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(thumbnailImage, 0, 0, width, height);

        // Replace the original image with the thumbnail
        previewImage.src = canvas.toDataURL(); // The thumbnail is in base64 format
    };


    var popup = document.getElementById('popup');
    popup.style.display = 'block';
    popup.style.left = (x + 10) + 'px';
    popup.style.top = (y + 10) + 'px';
}

// Function to hide the popup
function hidePreview() {
    var popup = document.getElementById('popup');
    popup.style.display = 'none';
}