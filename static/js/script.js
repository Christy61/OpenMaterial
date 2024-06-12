class BeforeAfter {
    constructor(entryObject) {
        const beforeAfterContainer = document.querySelector(entryObject.id);
        const before = beforeAfterContainer.querySelector('.bal-before');
        const after = beforeAfterContainer.querySelector('.bal-after');
        const handle = document.getElementById(entryObject.handle);
        const beforeLabel = beforeAfterContainer.querySelector('.beforeLabel');
        const afterLabel = beforeAfterContainer.querySelector('.afterLabel');

        const updateClipPaths = () => {
            const containerWidth = beforeAfterContainer.offsetWidth;
            const handlePos = handle.offsetLeft + handle.offsetWidth / 2;

            before.style.clipPath = `inset(0 ${containerWidth - handlePos}px 0 0)`;
            after.style.clipPath = `inset(0 0 0 ${handlePos}px)`;

            afterLabel.style.left = `${handlePos}px`;
        };

        const handleDrag = (handle, callback) => {
            let startX = 0;
            let startLeft = 0;

            const onMouseMove = (e) => {
                const containerRect = beforeAfterContainer.getBoundingClientRect();
                const newLeft = startLeft + (e.clientX - startX);

                if (newLeft > 0 && newLeft < containerRect.width) {
                    callback(newLeft);
                }
            };

            const onMouseUp = () => {
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
            };

            const onMouseDown = (e) => {
                startX = e.clientX;
                startLeft = handle.offsetLeft;
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            };

            handle.addEventListener('mousedown', onMouseDown);
        };

        handleDrag(handle, (newLeft) => {
            handle.style.left = `${newLeft - handle.offsetWidth / 2}px`;
            updateClipPaths();
        });

        beforeAfterContainer.addEventListener("touchstart", (e) => {
            const handle = e.target.closest('.bal-handle');
            if (!handle) return;

            const startX = e.touches[0].clientX;
            const startLeft = handle.offsetLeft;

            const onTouchMove = (e2) => {
                const containerRect = beforeAfterContainer.getBoundingClientRect();
                const newLeft = startLeft + (e2.touches[0].clientX - startX);

                if (newLeft > 0 && newLeft < containerRect.width) {
                    handle.style.left = `${newLeft - handle.offsetWidth / 2}px`;
                    updateClipPaths();
                }
            };

            const onTouchEnd = () => {
                beforeAfterContainer.removeEventListener('touchmove', onTouchMove);
                beforeAfterContainer.removeEventListener('touchend', onTouchEnd);
            };

            beforeAfterContainer.addEventListener('touchmove', onTouchMove);
            beforeAfterContainer.addEventListener('touchend', onTouchEnd);
        });

        window.addEventListener('resize', updateClipPaths);
        updateClipPaths();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    for (let i = 1; i <= 4; i++) {
        new BeforeAfter({
            id: `#example${i}`,
            handle: `handle1-${i}`
        });
    }
});

document.addEventListener('DOMContentLoaded', (event) => {
    document.querySelectorAll('.no-drag').forEach((img) => {
        img.ondragstart = () => false;
    });
});