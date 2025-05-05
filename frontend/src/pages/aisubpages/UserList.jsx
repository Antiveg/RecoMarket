import React, { useState, useEffect } from "react"
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import ai from '../../assets/ai.svg'
import UserAICard from "../../components/UserAICard"

/*
users: (user_id, user_latents)
products: product_id,product_name,aisle_id,department_id, (product_latents)
orders: order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order
order_products: order_id,product_id,add_to_cart_order,reordered
departments: department_id,department
aisles: aisle_id,aisle
*/

const temps = [
    {
        user_id: 1,
        user_name: 'Hello',
        user_img: "TEMPLATE USER",
        user_latents: [],
        order_count: 3
    },
    {
        user_id: 1,
        user_name: 'Hello',
        user_img: "TEMPLATE USER",
        user_latents: [],
        order_count: 3
    },
    {
        user_id: 1,
        user_name: 'Hello',
        user_img: "TEMPLATE USER",
        user_latents: [],
        order_count: 3
    },
    {
        user_id: 1,
        user_name: 'Hello',
        user_img: "TEMPLATE USER",
        user_latents: [],
        order_count: 3
    },
    {
        user_id: 1,
        user_name: 'Hello',
        user_img: "TEMPLATE USER",
        user_latents: [],
        order_count: 3
    },
]

const UserList = () => {

    const [loading, setLoading] = useState(false)
    const [users, setUsers] = useState([])
    const [error, setError] = useState('')

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true)
                const backend = import.meta.env.VITE_BACKEND_BASEURL
                const request = `${backend}/detailed-users`
                const response = await axios.get(request)
                setUsers(response.data.users)
            }catch(error){
                setError(error.response)
            }finally{
                setLoading(false)
            }
        }
        fetchData()
    }, [])

    if(loading){
        return (
            <div className='w-full h-full flex justify-center items-center text-base'>
                Fetching Users from Database ...
            </div>
        )
    }

    return (
        <div className='h-full w-full'>
            <p className='text-sm text-center text-gray-600 mb-4'><i>Disclaimer: This section only shows first 1000 users for efficiency. For more detail, please refer to <b>Evaluation section</b></i></p>
            <div className='h-auto rounded-5 grid lg:grid-cols-4 md:grid-cols-2 grid-rows-auto gap-2 rounded-xl'>
                {(users == null || users.length <= 0) ?
                    <p className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">No User yet ...</p> :
                    users.map((user) => (<UserAICard user={user} key={user.user_id}/>))
                }
            </div>
        </div>
    )
}

export default UserList