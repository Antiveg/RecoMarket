import React from "react";
import consumers from '../assets/consumers.svg'

const UserAICard = ({user}) => {

    const { user_id, user_name, user_img, order_count } = user

    const uploads = import.meta.env.VITE_UPLOADS_BASEURL
    const imgURL = `${uploads}/${user_img}`
    
    return (
        <div className="w-full h-auto flex flex-row p-2 gap-2 bg-[var(--surface)] rounded-md">
            <img src={imgURL} alt="No image yet..." className="w-0 flex-grow border-2 object-contain rounded-sm"/>
            <div className="flex flex-col w-3/4">
                <h3 className='font-bold text-base line-clamp-1'>{user_name}</h3>
                <p className='font-italic text-sm line-clamp-1'>User ID{user_id}</p>
                <p className='text-sm line-clamp-1'>Has Ordered {order_count} times</p>
            </div>
        </div>
    )
}

export default UserAICard