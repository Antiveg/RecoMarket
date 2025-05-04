import React from "react";
import ai from '../assets/ai.svg'

const ProductCard = ({product}) => {

    const { product_id, product_name, aisle_name, department_name, department_img, user_count } = product
    
    const uploads = import.meta.env.VITE_UPLOADS_BASEURL
    const imgURL = `${uploads}/${department_img}`

    return (
        <div className="w-full h-auto flex flex-row p-2 gap-2 bg-[var(--surface)] rounded-md">
            <img src={imgURL} alt="No image yet..." className="min-w-[60px] p-2 max-w-full w-1/4 object-contain rounded-sm"/>
            <div className="flex flex-col h-full w-3/4">
                <h3 className='font-bold text-md line-clamp-1'>{product_name}</h3>
                {department_name != null && <p className='text-sm line-clamp-1'>Department: {department_name}</p>}
                {aisle_name != null && <p className='text-sm line-clamp-1'>Aisle: {aisle_name}</p>}
                {user_count != null && <p className='text-sm line-clamp-1'>Has purchased by {user_count} consumer(s)</p>}
                <p className='font-italic text-right text-xs line-clamp-1 mt-2'>Product ID{product_id}</p>
            </div>
        </div>
    )
}

export default ProductCard